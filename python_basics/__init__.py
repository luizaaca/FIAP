"""Pacote `python_basics` — API leve exportada pelo pacote.

Aqui exportamos a classe do gerenciador de contexto e fornecemos
funções de acesso *lazy* aos arrays definidos em
`numpy_basic_array.py` para evitar executar código pesado/prints
de forma inesperada durante a importação do pacote.
"""

from .context_manager import GerenciadorDeContexto


# Lazy imports para evitar código pesado na importação do pacote
def get_y1():
    """Retorna `y1` definido em `numpy_basic_array.py` (import tardio)."""
    from .numpy_basic_array import y1

    return y1


# Lazy imports para evitar código pesado na importação do pacote
def get_y2():
    """Retorna `y2` definido em `numpy_basic_array.py` (import tardio)."""
    from .numpy_basic_array import y2

    return y2


def exemplo_operacao(*args, **kwargs):
    """Proxy para `numpy_basic_array.exemplo_operacao` que importa o
    submódulo somente quando a função é chamada (lazy import).
    """
    from .numpy_basic_array import exemplo_operacao as _ex

    return _ex(*args, **kwargs)


__all__ = ["GerenciadorDeContexto"]
