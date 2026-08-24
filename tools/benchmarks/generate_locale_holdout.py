#!/usr/bin/env python3
"""Generate frozen deterministic PT-BR locale dev/test panels.

The two panels contain no prompts from the original normal-QA benchmark. The
dev panel may be used to select a generic language contract; the test panel must
remain untouched until that contract is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def panel(prefix: str, variant: int) -> list[dict]:
    rows: list[dict] = []

    binary = [
        ("Todo número divisível por 4 é par?", "sim"),
        ("O número 27 é primo?", "não"),
        ("Um triângulo pode ter quatro lados?", "não"),
        ("A água congela a 0 °C sob pressão atmosférica padrão?", "sim"),
        ("O resultado de 9 vezes 7 é 63?", "sim"),
        ("O oceano Atlântico é maior que o Pacífico?", "não"),
        ("A soma de dois números ímpares é par?", "sim"),
        ("O Brasil fica na América do Norte?", "não"),
        ("Um byte normalmente contém oito bits?", "sim"),
        ("A raiz quadrada principal de 81 é 8?", "não"),
        ("HTTP 201 indica criação bem-sucedida de um recurso?", "sim"),
        ("Uma função injetiva precisa atribuir a mesma saída a entradas distintas?", "não"),
    ]
    if variant:
        binary = [
            ("Todo múltiplo de 10 termina em zero?", "sim"),
            ("O número 51 é primo?", "não"),
            ("Um pentágono possui cinco lados?", "sim"),
            ("O gelo é mais denso que a água líquida nas condições usuais?", "não"),
            ("O resultado de 12 vezes 6 é 72?", "sim"),
            ("A Lua é uma estrela?", "não"),
            ("O produto de dois números negativos é positivo?", "sim"),
            ("Portugal fica na América do Sul?", "não"),
            ("UTF-8 pode representar caracteres ASCII?", "sim"),
            ("A raiz quadrada principal de 144 é 14?", "não"),
            ("HTTP 204 normalmente não traz corpo de resposta?", "sim"),
            ("Uma lista vazia contém exatamente um elemento?", "não"),
        ]
    for index, (question, expected) in enumerate(binary, 1):
        rows.append({"id": f"{prefix}-b{index:02}", "category": "binary_pt",
                     "prompt": question + " Responda somente com Sim ou Não, em português.",
                     "grader": "exact_any", "expected": [expected]})

    lexical_a = [
        ("Qual é o nome em português do processo pelo qual plantas convertem luz em energia química?", ["fotossíntese"]),
        ("Como se chama em português o dispositivo que mede temperatura?", ["termômetro"]),
        ("Qual é o nome em português da camada gasosa que envolve a Terra?", ["atmosfera"]),
        ("Como se chama em português a ciência que estuda os seres vivos?", ["biologia"]),
        ("Qual é o nome em português do polígono com oito lados?", ["octógono"]),
        ("Como se chama em português a passagem direta do sólido para o gasoso?", ["sublimação"]),
        ("Qual é o nome em português do instrumento usado para observar astros distantes?", ["telescópio"]),
        ("Como se chama em português a unidade básica hereditária?", ["gene"]),
    ]
    lexical_b = [
        ("Qual é o nome em português do processo de divisão de uma célula em duas células idênticas?", ["mitose"]),
        ("Como se chama em português o aparelho que registra terremotos?", ["sismógrafo"]),
        ("Qual é o nome em português da linha que divide a Terra em hemisférios norte e sul?", ["equador", "linha do equador"]),
        ("Como se chama em português a ciência que estuda os astros?", ["astronomia"]),
        ("Qual é o nome em português do polígono com seis lados?", ["hexágono"]),
        ("Como se chama em português a passagem do gasoso para o líquido?", ["condensação"]),
        ("Qual é o nome em português do instrumento que mede pressão atmosférica?", ["barômetro"]),
        ("Como se chama em português a molécula que carrega a informação genética?", ["dna", "ácido desoxirribonucleico"]),
    ]
    for index, (question, expected) in enumerate(lexical_b if variant else lexical_a, 1):
        rows.append({"id": f"{prefix}-l{index:02}", "category": "lexical_pt",
                     "prompt": question + " Responda somente com o termo.",
                     "grader": "exact_any", "expected": expected})

    clarification_a = [
        ("Compare as opções para mim.", ["opções"]),
        ("Calcule o desconto da compra.", ["valor"]),
        ("Diga qual rota é mais rápida.", ["rotas"]),
        ("Escolha o melhor notebook.", ["notebooks"]),
        ("Resuma o documento.", ["documento"]),
        ("Verifique se a configuração está correta.", ["configuração"]),
        ("Qual plano devo contratar?", ["planos"]),
        ("Analise esses resultados.", ["resultados"]),
    ]
    clarification_b = [
        ("Qual desses produtos vale mais a pena?", ["produtos"]),
        ("Descubra o custo final para mim.", ["preços"]),
        ("Escolha o voo mais conveniente.", ["voos"]),
        ("Qual servidor devo comprar?", ["servidores"]),
        ("Explique o conteúdo do arquivo.", ["arquivo"]),
        ("Avalie se o desempenho está bom.", ["métricas"]),
        ("Qual assinatura é melhor?", ["assinaturas"]),
        ("Interprete esta comparação.", ["dados"]),
    ]
    for index, (request, required) in enumerate(clarification_b if variant else clarification_a, 1):
        rows.append({"id": f"{prefix}-c{index:02}", "category": "clarification_pt",
                     "prompt": request + " Faltam dados essenciais; faça uma única pergunta curta de esclarecimento em português.",
                     "grader": "pt_question", "required": required,
                     "forbidden": ["which", "what", "could you", "please", "do you"], "max_words": 14})

    reading_a = [
        ("Lia tinha 35 ingressos, vendeu 12 e recebeu mais 4. Quantos ingressos ela tem agora?", "27"),
        ("O trem saiu às 14h20 e chegou às 16h05. Quantos minutos durou a viagem?", "105"),
        ("Uma caixa contém 6 fileiras de 8 peças e 5 peças foram retiradas. Quantas restaram?", "43"),
        ("Rui leu 18 páginas na segunda, 27 na terça e 15 na quarta. Quantas páginas leu ao todo?", "60"),
        ("Um tanque de 90 litros estava com dois terços da capacidade. Quantos litros havia?", "60"),
        ("A reunião começou às 9h45 e terminou às 11h10. Quantos minutos durou?", "85"),
        ("Uma loja recebeu 72 canecas e as distribuiu igualmente em 9 caixas. Quantas por caixa?", "8"),
        ("Bia percorreu 3 km pela manhã e 2,5 km à tarde. Quantos quilômetros percorreu?", "5,5"),
    ]
    reading_b = [
        ("Caio tinha 48 moedas, gastou 19 e encontrou mais 7. Com quantas ficou?", "36"),
        ("O ônibus saiu às 8h35 e chegou às 10h10. Quantos minutos durou a viagem?", "95"),
        ("Um depósito tem 7 prateleiras com 9 caixas; 8 caixas foram removidas. Quantas restaram?", "55"),
        ("Nina escreveu 22 linhas de manhã, 16 à tarde e 12 à noite. Quantas linhas escreveu?", "50"),
        ("Um recipiente de 120 litros estava com três quartos da capacidade. Quantos litros havia?", "90"),
        ("A aula começou às 13h15 e terminou às 14h50. Quantos minutos durou?", "95"),
        ("Uma gráfica dividiu 96 cartazes igualmente em 12 pacotes. Quantos por pacote?", "8"),
        ("Davi caminhou 4,2 km cedo e 1,8 km depois. Quantos quilômetros caminhou?", "6"),
    ]
    for index, (question, expected) in enumerate(reading_b if variant else reading_a, 1):
        rows.append({"id": f"{prefix}-r{index:02}", "category": "reading_pt",
                     "prompt": question + " Responda somente com o número.",
                     "grader": "exact_any", "expected": [expected]})

    formats_a = [
        {"prompt": "Produza JSON com as chaves nome e ativo para nome Lia e ativo verdadeiro. Sem texto extra.",
         "grader": "json_exact", "expected": {"nome": "Lia", "ativo": True}},
        {"prompt": "Escreva exatamente duas linhas: primeira 'azul', segunda 'verde'.", "grader": "lines_exact", "expected": ["azul", "verde"]},
        {"prompt": "Produza JSON com as chaves cidade e quantidade para Recife e 3. Sem texto extra.",
         "grader": "json_exact", "expected": {"cidade": "Recife", "quantidade": 3}},
        {"prompt": "Escreva exatamente três linhas: primeira 'um', segunda 'dois', terceira 'três'.", "grader": "lines_exact", "expected": ["um", "dois", "três"]},
        {"prompt": "Produza JSON com as chaves idioma e codigo para português e pt-BR. Sem texto extra.",
         "grader": "json_exact", "expected": {"idioma": "português", "codigo": "pt-BR"}},
        {"prompt": "Escreva exatamente duas linhas: primeira 'entrada', segunda 'saída'.", "grader": "lines_exact", "expected": ["entrada", "saída"]},
        {"prompt": "Produza JSON com as chaves aprovado e nota para verdadeiro e 9. Sem texto extra.",
         "grader": "json_exact", "expected": {"aprovado": True, "nota": 9}},
        {"prompt": "Escreva exatamente três linhas: primeira 'norte', segunda 'sul', terceira 'leste'.", "grader": "lines_exact", "expected": ["norte", "sul", "leste"]},
    ]
    formats_b = [
        {"prompt": "Produza JSON com as chaves projeto e pronto para projeto Atlas e pronto falso. Sem texto extra.",
         "grader": "json_exact", "expected": {"projeto": "Atlas", "pronto": False}},
        {"prompt": "Escreva exatamente duas linhas: primeira 'claro', segunda 'escuro'.", "grader": "lines_exact", "expected": ["claro", "escuro"]},
        {"prompt": "Produza JSON com as chaves estado e total para Bahia e 7. Sem texto extra.",
         "grader": "json_exact", "expected": {"estado": "Bahia", "total": 7}},
        {"prompt": "Escreva exatamente três linhas: primeira 'alfa', segunda 'beta', terceira 'gama'.", "grader": "lines_exact", "expected": ["alfa", "beta", "gama"]},
        {"prompt": "Produza JSON com as chaves moeda e codigo para real e BRL. Sem texto extra.",
         "grader": "json_exact", "expected": {"moeda": "real", "codigo": "BRL"}},
        {"prompt": "Escreva exatamente duas linhas: primeira 'abrir', segunda 'fechar'.", "grader": "lines_exact", "expected": ["abrir", "fechar"]},
        {"prompt": "Produza JSON com as chaves valido e nivel para verdadeiro e 4. Sem texto extra.",
         "grader": "json_exact", "expected": {"valido": True, "nivel": 4}},
        {"prompt": "Escreva exatamente três linhas: primeira 'baixo', segunda 'médio', terceira 'alto'.", "grader": "lines_exact", "expected": ["baixo", "médio", "alto"]},
    ]
    for index, task in enumerate(formats_b if variant else formats_a, 1):
        rows.append({"id": f"{prefix}-f{index:02}", "category": "format_pt", **task})

    summaries_a = [
        ("A equipe adiou a entrega de sexta para segunda porque o teste de segurança falhou. Resuma em até 14 palavras, citando entrega, segunda e segurança.", ["entrega", "segunda", "segurança"]),
        ("O cliente relatou lentidão; o suporte aumentou a memória e o sistema normalizou em uma hora. Resuma em até 16 palavras, citando cliente, memória e normalizou.", ["cliente", "memória", "normalizou"]),
        ("A escola economizou 20% de energia após trocar lâmpadas e instalar sensores. Resuma em até 14 palavras, citando escola, energia e sensores.", ["escola", "energia", "sensores"]),
        ("O pedido chegou incompleto; a loja enviará os itens restantes amanhã sem custo. Resuma em até 15 palavras, citando pedido, amanhã e custo.", ["pedido", "amanhã", "custo"]),
    ]
    summaries_b = [
        ("A publicação mudou de terça para quinta porque faltava revisar as referências. Resuma em até 14 palavras, citando publicação, quinta e referências.", ["publicação", "quinta", "referências"]),
        ("A usuária perdeu acesso; o suporte redefiniu a senha e a conta voltou em dez minutos. Resuma em até 16 palavras, citando usuária, senha e voltou.", ["usuária", "senha", "voltou"]),
        ("O prédio reduziu 15% do consumo de água após consertar vazamentos e instalar medidores. Resuma em até 15 palavras, citando prédio, água e medidores.", ["prédio", "água", "medidores"]),
        ("A encomenda foi danificada; a transportadora enviará uma substituta amanhã sem cobrança. Resuma em até 15 palavras, citando encomenda, amanhã e cobrança.", ["encomenda", "amanhã", "cobrança"]),
    ]
    for index, (prompt, required) in enumerate(summaries_b if variant else summaries_a, 1):
        rows.append({"id": f"{prefix}-s{index:02}", "category": "summary_pt",
                     "prompt": prompt, "grader": "contains_all", "required": required,
                     "forbidden": ["the", "and", "will"], "max_words": 16})

    assert len(rows) == 48
    return rows


def write_panel(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, variant in (("dev", 0), ("test", 1)):
        path = args.outdir / f"locale_{name}_48.jsonl"
        digest = write_panel(path, panel(name, variant))
        print(f"{path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
