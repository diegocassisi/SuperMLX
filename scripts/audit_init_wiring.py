"""
[AI_DIRECTIVE]
ROL: Auditor de wiring de funciones init() para módulos de components.
OBJETIVO: Detectar definiciones def init() en components/*.py y verificar que estén cableadas en server.py.
ENTRADAS: Rutas a server.py y components/.
SALIDAS: Código de salida 0 si todo está cableado o debidamente justificado, 1 si falta cablear algún init().
REGLAS INVIOLABLES:
- Prohibido omitir módulos con init() no registrados como intencionalmente desconectados.
- Obligatorio verificación mediante AST para evitar falsos positivos de expresiones regulares.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Módulos intencionalmente desconectados (documentados en arquitectura)
KNOWN_DISCONNECTED_MODULES: Set[str] = {
    "cascade_routing",  # Desconectado a propósito en v2-refactor / Fase D
}


def find_modules_with_init(components_dir: Path) -> Dict[str, Path]:
    """Escanea components_dir y retorna un dict {module_name: file_path} de los que definen def init()."""
    modules_with_init: Dict[str, Path] = {}
    for py_file in sorted(components_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError as e:
            print(f"[ERROR] Error de sintaxis en {py_file.name}: {e}")
            sys.exit(1)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "init":
                modules_with_init[py_file.stem] = py_file
                break
    return modules_with_init


def find_init_calls_in_server(server_file: Path) -> Tuple[Set[str], Set[str]]:
    """
    Analiza server_file con AST para encontrar qué módulos o alias llaman a init().
    Retorna (modulos_llamados_directos_o_alias, funciones_init_llamadas).
    """
    content = server_file.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(server_file))

    # Mapear alias a nombres de módulo importados de components
    # Ej: from .components import adaptive_prefill as _ap_mod -> {'_ap_mod': 'adaptive_prefill'}
    # Ej: from .components.cache_lru import init as _cache_lru_init -> {'_cache_lru_init': 'cache_lru'}
    alias_to_module: Dict[str, str] = {}
    func_alias_to_module: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_path = node.module or ""
            if "components" in module_path:
                # Caso: from .components.module_name import init as foo_init
                parts = module_path.split(".")
                submod = parts[-1] if parts[-1] != "components" else None
                for alias in node.names:
                    imported_name = alias.name
                    as_name = alias.asname or imported_name
                    if submod:
                        if imported_name == "init":
                            func_alias_to_module[as_name] = submod
                    else:
                        alias_to_module[as_name] = imported_name

    called_modules: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Caso 1: _ap_mod.init(...)
            if isinstance(func, ast.Attribute) and func.attr == "init":
                if isinstance(func.value, ast.Name):
                    caller_var = func.value.id
                    if caller_var in alias_to_module:
                        called_modules.add(alias_to_module[caller_var])
            # Caso 2: _cache_lru_init(...)
            elif isinstance(func, ast.Name):
                called_func = func.id
                if called_func in func_alias_to_module:
                    called_modules.add(func_alias_to_module[called_func])

    return called_modules, set(func_alias_to_module.keys())


def audit_wiring(repo_root: Path) -> bool:
    """Ejecuta la auditoría completa de wiring de init() entre server.py y components."""
    server_file = repo_root / "supermlx" / "server.py"
    components_dir = repo_root / "supermlx" / "components"

    if not server_file.exists():
        print(f"[ERROR] No existe {server_file}")
        return False
    if not components_dir.exists():
        print(f"[ERROR] No existe {components_dir}")
        return False

    modules_with_init = find_modules_with_init(components_dir)
    called_modules, _ = find_init_calls_in_server(server_file)

    missing: List[str] = []
    wired: List[str] = []
    ignored: List[str] = []

    for mod in sorted(modules_with_init.keys()):
        if mod in KNOWN_DISCONNECTED_MODULES:
            ignored.append(mod)
        elif mod in called_modules:
            wired.append(mod)
        else:
            missing.append(mod)

    print("=" * 60)
    print(" AUDITORÍA DE WIRING INIT() — components → server.py")
    print("=" * 60)
    for mod in wired:
        print(f"  [OK] {mod}.init() conectado")
    for mod in ignored:
        print(f"  [DISCONNECTED INTENCIONAL] {mod}.init() (desconectado)")

    if missing:
        print("-" * 60)
        for mod in missing:
            print(f"  [FAIL] {mod}.init() DEFINIDO pero NO LLAMADO en server.py!")
        print("=" * 60)
        return False

    print("=" * 60)
    print(f"  RESULTADO: OK ({len(wired)} módulos conectados, {len(ignored)} intencionalmente desconectados)")
    print("=" * 60)
    return True


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    success = audit_wiring(root)
    sys.exit(0 if success else 1)
