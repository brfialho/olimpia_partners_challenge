# 🏦 Sistema de Pesquisa Automatizada de Empresas com Langchain - Fluxo de Implementação

## Arquitetura

Sistema de pesquisa automatizada que processa empresas em 4 etapas sequenciais. Usa LangChain para orquestrar um LLM customizado (Google Gemini) com APIs externas.

Cada etapa é independente com tratamento de erro isolado. Falhas parciais não interrompem o workflow.

---

## Pipeline de Execução

### 1. Resumo Executivo
- Função: `pesquisar_empresa_completa()`
- LangChain chain: `PromptTemplate | GeminiLLM`
- Prompt estruturado solicita: setor, histórico, produtos, posição de mercado
- Output: Texto de ~400 palavras

### 2. Notícias Recentes
- Função: `buscar_noticias()`
- Requisição HTTP para Google News RSS (pt-BR)
- Parse XML com ElementTree
- Extrai 3 itens: título, link, data
- Output: Array de dicionários

### 3. Identificação do Ticker
- Função: `buscar_ticker()`
- Chain LangChain com prompt few-shot
- Gemini retorna símbolo da ação (PETR4.SA, AAPL)
- Sanitização: remove markdown e espaços
- Output: String do ticker ou vazio

### 4. Cotação Atual
- Função: `obter_cotacao()`
- Requisição HTTP para Yahoo Finance Chart API
- Parse JSON para extrair preço, moeda, símbolo
- Depende do ticker da etapa anterior
- Output: Dict com valores ou zerados

---

## Componentes Técnicos

### GeminiLLM Class
Herda de `langchain_core.language_models.llms.LLM`:
- Implementa `_call()` para conectar com API do Gemini
- Config: temperatura 0.7, max 2048 tokens
- Try/catch retorna erro truncado em caso de falha

### LCEL Pattern
Composição declarativa com pipe operator:
```python
chain = PromptTemplate | LLM
chain.invoke({"empresa": nome})
```

### Orquestração
`executar_pesquisa()` chama as 4 etapas em sequência:
1. Captura resultados de cada função
2. Exibe no terminal com Colorama (headers, ícones, cores)
3. Pergunta se deseja salvar
4. Gera arquivo `.txt` em `./relatorios/` se confirmado

---

## Dependências Externas

**Google Gemini API**: Geração de texto (resumo e ticker). Requer chave no `.env`. Rate limit: 15 req/min.

**Google News RSS**: Notícias públicas sem autenticação. XML padrão.

**Yahoo Finance**: Cotações via API pública não oficial. Sem autenticação.

---

## 📦 Setup Rápido

### 1. Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Dependências

```bash
pip install -r requirements.txt
```

### 3. API Key (Gratuita)

Copie o arquivo de exemplo e adicione sua chave:

```bash
cp .env.example .env

```env
GOOGLE_API_KEY=sua_chave_aqui
```

**Obter chave**: https://aistudio.google.com/app/apikey

---

## 🚀 Uso

### Modo Interativo

```bash
python3 main.py
```

### Modo Direto

```bash
python3 main.py Petrobras
python3 main.py Apple
python3 main.py "Magazine Luiza"
```

---

## 📊 Empresas Suportadas

**Qualquer empresa de capital aberto**:

- **B3**: Petrobras, Vale, Itaú, Ambev, WEG, etc.
- **Internacional**: Apple, Microsoft, Google, Tesla, etc.


---

## 💾 Salvamento de Relatórios

Ao final da execução, o sistema pergunta se deseja salvar:

```
Deseja salvar este relatório? (s/n):
```

Relatórios salvos em: `./relatorios/[empresa].txt`
