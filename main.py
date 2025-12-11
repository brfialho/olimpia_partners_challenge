#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Pesquisa Automatizada de Empresas
LangChain + Google Gemini 2.5 Flash
Autor: Bruno Fialho
Data: 11/12/2025
"""

# ==================== IMPORTS ====================
import os
import sys
import warnings
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, List, Optional

from dotenv import load_dotenv
from colorama import Fore, Style, init

import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Configurações iniciais
warnings.filterwarnings('ignore')
init(autoreset=True)
load_dotenv()


# ==================== CLASSES ====================

class GeminiLLM(LLM):
    """LLM customizado integrado ao LangChain usando Google Gemini"""
    
    api_key: str
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Chama a API do Google Gemini"""
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                )
            )
            
            return response.text.strip()
        except Exception as e:
            return f"Erro na geração: {str(e)[:200]}"


# ==================== FUNÇÕES AUXILIARES ====================

def print_header(text: str):
    """Imprime cabeçalho formatado"""
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.CYAN}{text.center(80)}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")


def print_section(title: str):
    """Imprime título de seção"""
    print(f"\n{Fore.YELLOW}▶ {title}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'-' * 80}{Style.RESET_ALL}")


def get_llm() -> GeminiLLM:
    """Retorna instância configurada do LLM"""
    return GeminiLLM(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model_name="gemini-2.5-flash",
        temperature=0.7
    )


# ==================== FUNÇÕES DE PESQUISA ====================

def pesquisar_empresa_completa(nome_empresa: str) -> dict:
    """A. Gera resumo executivo da empresa usando LangChain + Gemini"""
    
    llm = get_llm()
    
    template = """Você é um analista financeiro especializado em Investment Banking.

Forneça um RESUMO EXECUTIVO PROFISSIONAL sobre a empresa: {empresa}

Inclua obrigatoriamente:

1. SETOR DE ATUAÇÃO: Qual o principal setor e indústria
2. BREVE HISTÓRICO: Fundação, evolução, marcos importantes
3. PRINCIPAIS PRODUTOS/SERVIÇOS: O que a empresa oferece
4. POSIÇÃO NO MERCADO: Relevância e competitividade

Responda em português brasileiro de forma objetiva e estruturada (máximo 500 palavras)."""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    try:
        resultado = chain.invoke({"empresa": nome_empresa})
        texto = resultado.strip() if isinstance(resultado, str) else str(resultado).strip()
        
        return {
            "sucesso": True,
            "analise": texto if texto else "Análise gerada com sucesso.",
            "empresa": nome_empresa
        }
    except Exception as e:
        return {
            "sucesso": False,
            "analise": f"Erro ao processar: {str(e)}",
            "empresa": nome_empresa
        }


def buscar_noticias(empresa: str) -> list:
    """B. Busca notícias reais via Google News RSS"""
    
    try:
        query = urllib.parse.quote(empresa)
        url = f"https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        noticias = []
        
        for item in root.findall(".//item")[:5]:
            titulo = item.find("title")
            link = item.find("link")
            pub_date = item.find("pubDate")
            
            if titulo is not None and link is not None:
                noticia = {
                    "titulo": titulo.text.strip() if titulo.text else "Sem título",
                    "link": link.text.strip() if link.text else "",
                    "data": pub_date.text.strip() if pub_date is not None and pub_date.text else "Data não disponível"
                }
                noticias.append(noticia)
        
        return noticias
        
    except Exception as e:
        print(f"{Fore.RED}Erro ao buscar notícias: {str(e)}{Style.RESET_ALL}")
        return []


def buscar_ticker(empresa: str) -> str:
    """C. Identifica o ticker da ação usando LangChain + Gemini"""
    
    llm = get_llm()
    
    template = """Você é um especialista em mercado de ações.

Identifique o TICKER (símbolo da ação) da empresa: {empresa}

Se for empresa brasileira, forneça o ticker da B3 (ex: PETR4.SA, VALE3.SA, ITUB4.SA)
Se for empresa internacional, forneça o ticker da bolsa principal (ex: AAPL, MSFT, GOOGL)

Responda APENAS com o ticker, sem explicações.

Exemplos:
- Petrobras → PETR4.SA
- Vale → VALE3.SA
- Apple → AAPL
- Microsoft → MSFT

Ticker da empresa {empresa}:"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    try:
        resultado = chain.invoke({"empresa": empresa})
        ticker = resultado.strip() if isinstance(resultado, str) else str(resultado).strip()
        ticker = ticker.replace('*', '').replace('`', '').split()[0] if ticker else ""
        
        return ticker
    except:
        return ""


def obter_cotacao(ticker: str) -> dict:
    """Obtém cotação atual da ação via Yahoo Finance"""
    
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        result = data.get("chart", {}).get("result", [{}])[0]
        meta = result.get("meta", {})
        
        return {
            "ticker": ticker,
            "preco": meta.get("regularMarketPrice", 0),
            "moeda": meta.get("currency", "USD"),
            "nome": meta.get("symbol", ticker)
        }
    except:
        return {"ticker": ticker, "preco": 0, "moeda": "N/A", "nome": ticker}


# ==================== FUNÇÕES DE RELATÓRIO ====================

def salvar_relatorio(nome_empresa: str, resumo: dict, noticias: list, ticker: str, cotacao: dict) -> str:
    """Salva o relatório completo em arquivo .txt"""
    
    os.makedirs("relatorios", exist_ok=True)
    
    nome_arquivo = nome_empresa.replace(" ", "_").replace("/", "-")
    caminho = f"relatorios/{nome_arquivo}.txt"
    
    conteudo = f"""================================================================================
RELATÓRIO DE PESQUISA AUTOMATIZADA: {nome_empresa.upper()}
================================================================================

Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Sistema: LangChain + Google Gemini 2.5 Flash

================================================================================
A. RESUMO/DESCRIÇÃO DA EMPRESA
================================================================================

{resumo.get('analise', 'Não disponível')}

================================================================================
B. ÚLTIMAS NOTÍCIAS RELEVANTES
================================================================================

"""
    
    if noticias:
        for i, noticia in enumerate(noticias, 1):
            conteudo += f"[{i}] {noticia.get('titulo', f'Notícia {i}')}\n"
            if noticia.get('data'):
                conteudo += f"    Data: {noticia['data']}\n"
            if noticia.get('link'):
                conteudo += f"    Link: {noticia['link']}\n"
            conteudo += "\n"
    else:
        conteudo += "Nenhuma notícia disponível.\n\n"
    
    conteudo += f"""================================================================================
C. VALOR DA AÇÃO (COTAÇÃO ATUAL)
================================================================================

"""
    
    if ticker:
        conteudo += f"Ticker: {ticker}\n"
        if cotacao.get("preco", 0) > 0:
            conteudo += f"Preço Atual: {cotacao['moeda']} {cotacao['preco']:.2f}\n"
        else:
            conteudo += "Cotação não disponível.\n"
    else:
        conteudo += "Ticker não encontrado.\n"
    
    conteudo += f"""
================================================================================
Relatório gerado automaticamente via LangChain + Google Gemini
================================================================================
"""
    
    try:
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        return caminho
    except Exception as e:
        print(f"{Fore.RED}Erro ao salvar relatório: {str(e)}{Style.RESET_ALL}")
        return None


def executar_pesquisa(nome_empresa: str):
    """Executa workflow completo de pesquisa"""
    
    print_header(f"PESQUISA AUTOMATIZADA: {nome_empresa.upper()}")
    print(f"{Fore.WHITE}Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}{Style.RESET_ALL}")
    
    # Variáveis para salvar dados
    resultado_resumo = {}
    noticias = []
    ticker = ""
    cotacao = {}
    
    # ========== A. RESUMO DA EMPRESA ==========
    print_section("📋 A. RESUMO/DESCRIÇÃO DA EMPRESA")
    print(f"{Fore.MAGENTA}🤖 LangChain Chain executando com Google Gemini...{Style.RESET_ALL}\n")
    
    resultado_resumo = pesquisar_empresa_completa(nome_empresa)
    
    if resultado_resumo["sucesso"]:
        print(f"{Fore.WHITE}{resultado_resumo['analise']}{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.RED}Erro: {resultado_resumo['analise']}{Style.RESET_ALL}\n")
    
    # ========== B. NOTÍCIAS RECENTES ==========
    print_section("📰 B. ÚLTIMAS NOTÍCIAS RELEVANTES")
    print(f"{Fore.MAGENTA}🔍 Buscando notícias reais via Google News...{Style.RESET_ALL}\n")
    
    noticias = buscar_noticias(nome_empresa)
    
    if noticias:
        print(f"{Fore.GREEN}✓ {len(noticias)} notícias encontradas:{Style.RESET_ALL}\n")
        
        for i, noticia in enumerate(noticias, 1):
            print(f"{Fore.WHITE}[{i}] {noticia.get('titulo', f'Notícia {i}')}")
            if noticia.get('data'):
                print(f"    Data: {noticia['data']}")
            if noticia.get('link'):
                print(f"    Link: {noticia['link']}")
            print(f"{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠ Não foi possível buscar notícias{Style.RESET_ALL}\n")
    
    # ========== C. TICKER E COTAÇÃO ==========
    print_section("💰 C. VALOR DA AÇÃO (COTAÇÃO ATUAL)")
    print(f"{Fore.MAGENTA}🤖 LangChain Chain identificando ticker com Google Gemini...{Style.RESET_ALL}\n")
    
    ticker = buscar_ticker(nome_empresa)
    
    if ticker:
        print(f"{Fore.GREEN}✓ Ticker encontrado: {ticker}{Style.RESET_ALL}\n")
        
        cotacao = obter_cotacao(ticker)
        
        if cotacao["preco"] > 0:
            print(f"{Fore.WHITE}Ticker:       {cotacao['ticker']}")
            print(f"Preço Atual:  {cotacao['moeda']} {cotacao['preco']:.2f}{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.YELLOW}⚠ Não foi possível obter cotação para {ticker}{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.YELLOW}⚠ Ticker não encontrado automaticamente{Style.RESET_ALL}\n")
    
    # ========== SALVAR RELATÓRIO ==========
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    resposta = input(f"{Fore.YELLOW}Deseja salvar este relatório? (s/n): {Style.RESET_ALL}").strip().lower()
    
    if resposta in ['s', 'sim', 'y', 'yes']:
        caminho = salvar_relatorio(nome_empresa, resultado_resumo, noticias, ticker, cotacao)
        if caminho:
            print(f"{Fore.GREEN}✓ Relatório salvo em: {caminho}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ Não foi possível salvar o relatório{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Relatório não salvo.{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")


# ==================== MAIN ====================

def main():
    """Função principal"""
    
    if len(sys.argv) > 1:
        nome_empresa = " ".join(sys.argv[1:])
    else:
        print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗")
        print(f"{Fore.CYAN}║{'SISTEMA DE PESQUISA AUTOMATIZADA DE EMPRESAS'.center(78)}║")
        print(f"{Fore.CYAN}║{'LangChain + Google Gemini 2.5 Flash'.center(78)}║")
        print(f"{Fore.CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")
        
        nome_empresa = input(f"{Fore.YELLOW}Digite o nome da empresa: {Style.RESET_ALL}").strip()
        
        if not nome_empresa:
            print(f"{Fore.RED}✗ Nome da empresa não pode estar vazio{Style.RESET_ALL}")
            sys.exit(1)
    
    executar_pesquisa(nome_empresa)


if __name__ == "__main__":
    main()
