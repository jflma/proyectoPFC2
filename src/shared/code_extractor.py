import re
from typing import Optional


def extract_code_block(llm_response: str, expected_language: str = "") -> Optional[str]:
    """
    Extrae el primer bloque de código de una respuesta de LLM.
    Maneja variantes: ```python, ```cpp, ```, ``` (sin lenguaje).

    Retorna el código limpio o None si no se encontró ningún bloque.

    REGLA R-06: NO usar directamente response.split('```')[1].
    """
    patrones = [
        rf"```{re.escape(expected_language)}\n(.*?)```",  # Con lenguaje específico
        r"```\w*\n(.*?)```",                               # Con cualquier lenguaje
        r"```(.*?)```",                                    # Sin salto de línea
    ]
    for patron in patrones:
        match = re.search(patron, llm_response, re.DOTALL)
        if match:
            codigo = match.group(1).strip()
            if len(codigo) > 10:  # Descartar bloques trivialmente vacíos
                return codigo
    return None


def extraer_bloques_codigo_markdown(texto: str) -> list[tuple[str, str]]:
    """
    Extrae bloques de código delimitados por ``` en texto Markdown.
    Retorna lista de (lenguaje_declarado, contenido).
    Si el lenguaje no está declarado, retorna ("unknown", contenido).
    """
    patron = r"```(\w*)\n(.*?)```"
    matches = re.findall(patron, texto, re.DOTALL)
    return [(lang.lower() or "unknown", code.strip()) for lang, code in matches]
