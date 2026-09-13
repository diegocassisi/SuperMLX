import time
import mlx.core as mx

def benchmark_real_sync():
    vocab_size = 248000
    steps = 150

    # Crear tensores iniciales
    logits = mx.random.normal(shape=(1, vocab_size))
    mx.eval(logits)

    # 1. Pipeline puro sin sync (cola asíncrona de Metal fluyendo)
    mx.metal.clear_cache()
    l = logits
    start = time.perf_counter()
    tokens = []
    for _ in range(steps):
        probs = mx.softmax(l, axis=-1)
        tok = mx.argmax(probs, axis=-1)
        tokens.append(tok)
        l = l + 0.001
    mx.eval(tokens)
    t_async = time.perf_counter() - start

    # 2. Pipeline forzando .item() (sincronización CPU/GPU obligatoria cada token)
    mx.metal.clear_cache()
    l = logits
    start = time.perf_counter()
    tokens_sync = []
    sync_delays = []
    for _ in range(steps):
        probs = mx.softmax(l, axis=-1)
        log_probs = mx.log(probs + 1e-10)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        
        # SINCRONIZACIÓN BLOQUEANTE:
        # Obliga a Metal a vaciar los command buffers y parar la CPU hasta que la GPU termine
        t_before_sync = time.perf_counter()
        val = entropy.item()
        t_sync_cost = time.perf_counter() - t_before_sync
        sync_delays.append(t_sync_cost)

        if val > 1.0:
            l = l * 1.001
        tok = mx.argmax(probs, axis=-1)
        tokens_sync.append(tok)
    mx.eval(tokens_sync)
    t_sync = time.perf_counter() - start

    # 3. Pipeline vectorizado con mx.where (cero llamadas a .item(), el grafo se resuelve dentro de GPU)
    mx.metal.clear_cache()
    l = logits
    start = time.perf_counter()
    tokens_vec = []
    for _ in range(steps):
        probs = mx.softmax(l, axis=-1)
        log_probs = mx.log(probs + 1e-10)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        # Vectorizado en GPU pura:
        mask = mx.where(entropy > 1.0, 1.001, 1.0)
        l = l * mask
        tok = mx.argmax(probs, axis=-1)
        tokens_vec.append(tok)
    mx.eval(tokens_vec)
    t_vec = time.perf_counter() - start

    print("=" * 65)
    print("RESULTADOS EMPÍRICOS (Vocabulario = 248.000 tokens):")
    print("=" * 65)
    print(f"1. Asíncrono puro (Baseline):          {t_async*1000/steps:.3f} ms/token ({steps/t_async:.1f} TPS)")
    print(f"2. Con Sync en Python (.item()):       {t_sync*1000/steps:.3f} ms/token ({steps/t_sync:.1f} TPS)")
    print(f"   -> Costo promedio del .item():      {sum(sync_delays)*1000/steps:.3f} ms/token parado esperando Metal")
    print(f"3. Vectorizado puro Metal (mx.where):  {t_vec*1000/steps:.3f} ms/token ({steps/t_vec:.1f} TPS)")
    print("=" * 65)
    print(f"Sobrecosto real del .item():           +{(t_sync - t_async)/t_async * 100:.1f}%")
    print(f"Sobrecosto de mx.where vectorizado:    +{(t_vec - t_async)/t_async * 100:.1f}%")
    print("=" * 65)

if __name__ == "__main__":
    benchmark_real_sync()
