// CoreML draft model server for speculative decoding.
// Loads Qwen3.5-0.8B on ANE with zero-copy state management.
// Protocol: line-based stdin/stdout.
//   LOAD <path>           → loads mlpackage, prints READY
//   FEED <token_id> <pos> → feeds token, prints predicted next token id
//   DRAFT <token_id> <pos> <k> → feeds token, generates k draft tokens, prints one per line, then DONE
//   RESET                 → zeros all states, prints OK
//   QUIT                  → exits

import CoreML
import Foundation

// ─── Config ───────────────────────────────────────────────────────────
let numLayers = 24
let rotaryDim = 64
let half = rotaryDim / 2
let ropeTheta: Float = 10_000_000.0
let vocab = 248320

// ─── RoPE tables ──────────────────────────────────────────────────────
func buildRope(maxSeq: Int, isCos: Bool) -> [Float] {
    var out = [Float](repeating: 0, count: maxSeq * rotaryDim)
    for p in 0..<maxSeq {
        for i in 0..<half {
            let theta = powf(ropeTheta, Float(-2 * i) / Float(rotaryDim))
            let a = Float(p) * theta
            let v = isCos ? cosf(a) : sinf(a)
            out[p * rotaryDim + i]        = v
            out[p * rotaryDim + i + half] = v
        }
    }
    return out
}

// ─── Zero-copy state feature provider ─────────────────────────────────
// Passes previous output's new_state_X_Y directly as current input's state_X_Y
// without copying through CPU memory.
final class DecodeFeatures: NSObject, MLFeatureProvider {
    let fvTok: MLFeatureValue
    let fvPos: MLFeatureValue
    let fvCos: MLFeatureValue
    let fvSin: MLFeatureValue
    let prevOut: MLFeatureProvider?
    let initialStates: [String: MLFeatureValue]?
    let rename: [String: String]  // state_X_Y -> new_state_X_Y
    let featureNames: Set<String>
    
    init(tok: MLFeatureValue, pos: MLFeatureValue,
         cos: MLFeatureValue, sin: MLFeatureValue,
         prevOut: MLFeatureProvider?,
         initialStates: [String: MLFeatureValue]?,
         stateIn: [String], stateOut: [String]) {
        self.fvTok = tok; self.fvPos = pos
        self.fvCos = cos; self.fvSin = sin
        self.prevOut = prevOut
        self.initialStates = initialStates
        var r: [String: String] = [:]
        for (a, b) in zip(stateIn, stateOut) { r[a] = b }
        self.rename = r
        var names: Set<String> = ["input_token", "position", "cos", "sin"]
        for n in stateIn { names.insert(n) }
        self.featureNames = names
        super.init()
    }
    
    func featureValue(for name: String) -> MLFeatureValue? {
        switch name {
        case "input_token": return fvTok
        case "position":    return fvPos
        case "cos":         return fvCos
        case "sin":         return fvSin
        default:
            if let prev = prevOut, let outName = rename[name] {
                return prev.featureValue(for: outName)
            }
            return initialStates?[name]
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────
var model: MLModel?
var cosTable: [Float] = []
var sinTable: [Float] = []
var maxSeq = 128
var stateIn: [String] = []
var stateOut: [String] = []
var initialStates: [String: MLFeatureValue] = [:]
var lastOutput: MLFeatureProvider? = nil

// Reusable arrays
var arrTok: MLMultiArray!
var arrPos: MLMultiArray!
var arrCos: MLMultiArray!
var arrSin: MLMultiArray!
var fvTok: MLFeatureValue!
var fvPos: MLFeatureValue!
var fvCos: MLFeatureValue!
var fvSin: MLFeatureValue!

func setupReusableArrays() {
    arrTok = try! MLMultiArray(shape: [1, 1], dataType: .int32)
    arrPos = try! MLMultiArray(shape: [1], dataType: .float32)
    arrCos = try! MLMultiArray(shape: [1, 1, NSNumber(value: rotaryDim)], dataType: .float16)
    arrSin = try! MLMultiArray(shape: [1, 1, NSNumber(value: rotaryDim)], dataType: .float16)
    fvTok = MLFeatureValue(multiArray: arrTok)
    fvPos = MLFeatureValue(multiArray: arrPos)
    fvCos = MLFeatureValue(multiArray: arrCos)
    fvSin = MLFeatureValue(multiArray: arrSin)
}

func float32ToFloat16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = (bits >> 16) & 0x8000
    let exp = Int((bits >> 23) & 0xFF) - 127 + 15
    let frac = bits & 0x7FFFFF
    if exp <= 0 { return UInt16(sign) }
    if exp >= 31 { return UInt16(sign | 0x7C00) }
    return UInt16(sign | UInt32(exp << 10) | (frac >> 13))
}

func fillRope(pos: Int) {
    let clampedPos = min(pos, maxSeq - 1)
    let offset = clampedPos * rotaryDim
    arrPos[0] = NSNumber(value: Float(pos))
    let cosPtr = arrCos.dataPointer.assumingMemoryBound(to: UInt16.self)
    let sinPtr = arrSin.dataPointer.assumingMemoryBound(to: UInt16.self)
    for i in 0..<rotaryDim {
        cosPtr[i] = float32ToFloat16(cosTable[offset + i])
        sinPtr[i] = float32ToFloat16(sinTable[offset + i])
    }
}

func fillToken(_ tokenId: Int32) {
    arrTok[0] = NSNumber(value: tokenId)
}

func predict(tokenId: Int32, pos: Int) -> MLFeatureProvider {
    fillToken(tokenId)
    fillRope(pos: pos)
    let features = DecodeFeatures(
        tok: fvTok, pos: fvPos, cos: fvCos, sin: fvSin,
        prevOut: lastOutput,
        initialStates: lastOutput == nil ? initialStates : nil,
        stateIn: stateIn, stateOut: stateOut)
    let out = try! model!.prediction(from: features)
    lastOutput = out
    return out
}

func argmax(from output: MLFeatureProvider) -> Int32 {
    let logits = output.featureValue(for: "logits")!.multiArrayValue!
    var bestIdx: Int32 = 0
    var bestVal: Float = -Float.infinity
    let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
    for i in 0..<vocab {
        let v = ptr[i]
        if v > bestVal { bestVal = v; bestIdx = Int32(i) }
    }
    return bestIdx
}

var savedOutput: MLFeatureProvider? = nil
var savedPos: Int = 0

func resetStates() {
    lastOutput = nil
}

func saveState(pos: Int) {
    // Deep copy state by reading all feature values into fresh MLMultiArrays
    guard let prev = lastOutput else {
        savedOutput = nil
        savedPos = pos
        return
    }
    var copied: [String: MLFeatureValue] = [:]
    for outName in stateOut {
        if let fv = prev.featureValue(for: outName),
           let arr = fv.multiArrayValue {
            let copy = try! MLMultiArray(shape: arr.shape, dataType: arr.dataType)
            let bytesPerElement = arr.dataType == .float32 ? 4 : 2
            memcpy(copy.dataPointer, arr.dataPointer, arr.count * bytesPerElement)
            copied[outName] = MLFeatureValue(multiArray: copy)
        }
    }
    // Also copy logits
    if let fv = prev.featureValue(for: "logits"),
       let arr = fv.multiArrayValue {
        let copy = try! MLMultiArray(shape: arr.shape, dataType: arr.dataType)
        memcpy(copy.dataPointer, arr.dataPointer, arr.count * 4)  // float32
        copied["logits"] = MLFeatureValue(multiArray: copy)
    }
    savedOutput = DictFeatureProvider(dict: copied)
    savedPos = pos
}

func restoreState() {
    lastOutput = savedOutput
}

// Simple feature provider backed by a dictionary
final class DictFeatureProvider: NSObject, MLFeatureProvider {
    let dict: [String: MLFeatureValue]
    var featureNames: Set<String> { Set(dict.keys) }
    init(dict: [String: MLFeatureValue]) { self.dict = dict; super.init() }
    func featureValue(for name: String) -> MLFeatureValue? { dict[name] }
}

func loadModel(path: String) {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuOnly
    let url = URL(fileURLWithPath: path)
    
    // Compile mlpackage → mlmodelc
    let compiledURL: URL
    if path.hasSuffix(".mlmodelc") {
        compiledURL = url
    } else {
        fputs("Compiling model (first time may take a while)...\n", stderr)
        compiledURL = try! MLModel.compileModel(at: url)
        fputs("Compiled to: \(compiledURL.path)\n", stderr)
    }
    model = try! MLModel(contentsOf: compiledURL, configuration: cfg)
    
    // Build state names
    stateIn = []; stateOut = []
    for i in 0..<numLayers {
        stateIn.append("state_\(i)_a"); stateIn.append("state_\(i)_b")
        stateOut.append("new_state_\(i)_a"); stateOut.append("new_state_\(i)_b")
    }
    
    // Build zero-initialized states from model spec
    let spec = model!.modelDescription
    initialStates = [:]
    for name in stateIn {
        if let desc = spec.inputDescriptionsByName[name] {
            let constraint = desc.multiArrayConstraint!
            let shape = constraint.shape
            let dt = constraint.dataType
            let arr = try! MLMultiArray(shape: shape, dataType: dt)
            // zero-fill (MLMultiArray is zero-initialized)
            initialStates[name] = MLFeatureValue(multiArray: arr)
        }
    }
    
    // Detect maxSeq from state shape
    for name in stateIn {
        if let desc = spec.inputDescriptionsByName[name] {
            let shape = desc.multiArrayConstraint!.shape.map { $0.intValue }
            if shape.count == 4 && shape[1] == 2 { // KV cache shape (1,2,maxseq,head_dim)
                maxSeq = shape[2]
                break
            }
        }
    }
    
    cosTable = buildRope(maxSeq: maxSeq, isCos: true)
    sinTable = buildRope(maxSeq: maxSeq, isCos: false)
    setupReusableArrays()
    lastOutput = nil
}

// ─── Command loop ─────────────────────────────────────────────────────
setbuf(stdout, nil)  // unbuffered output

while let line = readLine() {
    let parts = line.split(separator: " ")
    guard !parts.isEmpty else { continue }
    
    switch parts[0] {
    case "LOAD":
        let path = String(parts[1])
        loadModel(path: path)
        print("READY \(maxSeq)")
        
    case "FEED":
        let tokenId = Int32(parts[1])!
        let pos = Int(parts[2])!
        let out = predict(tokenId: tokenId, pos: pos)
        let next = argmax(from: out)
        print(next)
        
    case "DRAFT":
        let tokenId = Int32(parts[1])!
        var pos = Int(parts[2])!
        let k = Int(parts[3])!
        var current = tokenId
        // Feed the seed token first
        var out = predict(tokenId: current, pos: pos)
        pos += 1
        // Generate k draft tokens
        for _ in 0..<k {
            let next = argmax(from: out)
            print(next)
            out = predict(tokenId: next, pos: pos)
            pos += 1
            current = next
        }
        print("DONE")
        
    case "RESET":
        resetStates()
        print("OK")
        
    case "SAVE":
        let pos = Int(parts[1])!
        saveState(pos: pos)
        print("OK")
        
    case "RESTORE":
        restoreState()
        print("OK \(savedPos)")
        
    case "QUIT":
        exit(0)
        
    default:
        print("ERROR unknown command: \(parts[0])")
    }
}
