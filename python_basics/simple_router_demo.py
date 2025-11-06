"""
Exemplo de classe decorator para rotas, inspirado no FastAPI.
Suporta múltiplos métodos HTTP, parâmetros dinâmicos e validação básica.
"""

import re


class SimpleRouter:
    def __init__(self):
        # Dicionário: método -> lista de rotas (cada rota é um dict)
        self.routes = {"GET": [], "POST": []}

    def get(self, path):
        return self._add_route("GET", path)

    def post(self, path):
        return self._add_route("POST", path)

    def _add_route(self, method, path):
        def decorator(func):
            # Salva rota, função e path regex para parâmetros dinâmicos
            param_names = re.findall(r"{(.*?)}", path)  # Extrai nomes dos parâmetros
            # Converte o caminho da rota com parâmetros dinâmicos (ex: '/users/{user_id}')
            # em uma expressão regular com grupos nomeados (ex: '/users/(?P<user_id>[^/]+)')
            # Isso permite extrair automaticamente os valores dos parâmetros da URL.
            path_regex = re.sub(r"{(.*?)}", r"(?P<\1>[^/]+)", path)
            self.routes[method].append(
                {
                    "path": path,
                    "func": func,
                    "param_names": param_names,
                    "path_regex": re.compile(f"^{path_regex}$"),
                }
            )
            return func

        return decorator

    def call_route(self, method, path, body=None):
        # Busca rota correspondente ao método e path
        for route in self.routes.get(method, []):
            match = route["path_regex"].match(path)
            if match:
                # Extrai parâmetros dinâmicos da URL
                params = match.groupdict()
                # Validação simples: se for POST, espera body não vazio
                if method == "POST" and not body:
                    raise ValueError("Body obrigatório para POST")
                # Chama função com parâmetros dinâmicos e body (se houver)
                return (
                    route["func"](**params, body=body)
                    if body
                    else route["func"](**params)
                )
        raise ValueError(f"Rota '{path}' com método '{method}' não encontrada.")


# Instancia o router
router = SimpleRouter()


# GET com parâmetro dinâmico
@router.get("/users/{user_id}")
def get_user(user_id):
    return {"id": int(user_id), "name": "Maria"}


# POST com parâmetro dinâmico e body
@router.post("/users/{user_id}")
def update_user(user_id, body):
    # Validação simples do body
    if not isinstance(body, dict) or "name" not in body:
        raise ValueError("Body deve conter o campo 'name'")
    return {"id": int(user_id), "name": body["name"], "status": "atualizado"}


# POST sem parâmetro dinâmico
@router.post("/produtos")
def create_produto(body):
    if not isinstance(body, dict) or "nome" not in body:
        raise ValueError("Body deve conter o campo 'nome'")
    return {"id": 99, "nome": body["nome"], "status": "criado"}


# Testes das rotas registradas
if __name__ == "__main__":
    print("GET /users/1:")
    print(router.call_route("GET", "/users/1"))  # {'id': 1, 'name': 'Maria'}

    print("\nPOST /users/1 com body válido:")
    print(
        router.call_route("POST", "/users/1", body={"name": "João"})
    )  # {'id': 1, 'name': 'João', 'status': 'atualizado'}

    print("\nPOST /produtos com body válido:")
    print(
        router.call_route("POST", "/produtos", body={"nome": "Caderno"})
    )  # {'id': 99, 'nome': 'Caderno', 'status': 'criado'}

    print("\nPOST /users/1 com body inválido:")
    try:
        router.call_route("POST", "/users/1", body={})
    except ValueError as e:
        print("Erro capturado:", e)

    print("\nGET /users/abc (parâmetro string inválido):")
    try:
        router.call_route("GET", "/users/abc")
    except ValueError as e:
        print("Erro capturado:", e)

    print("\nRota inexistente:")
    try:
        router.call_route("GET", "/nao_existe")
    except ValueError as e:
        print("Erro capturado:", e)
