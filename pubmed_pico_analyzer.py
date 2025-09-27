import time
import pandas as pd
import google.generativeai as genai
from google.colab import userdata
from Bio import Entrez
from tqdm import tqdm
import xml.etree.ElementTree as ET

# 상수 정의
BATCH_SIZE = 200
SLEEP_INTERVAL = 0.4  # PubMed API 호출 사이의 간격 (초)
MAX_RETMAX = 10000  # PubMed에서 한 번에 검색할 최대 논문 수
MIN_ABSTRACT_LENGTH = 50  # 분석을 위한 최소 초록 길이

# Gemini 프롬프트 템플릿
PROMPT_TEMPLATE = """
당신은 임상 연구 논문을 분석하는 전문가입니다. 아래 제공된 초록(Abstract)을 읽고, 다음 PICO 기준에 부합하는 Original Article인지 판단해주세요.
- Population: {p}, Intervention: {i}, Control: {c}, Outcome: {o}
판단 기준: "이 논문이 [ {p} ]을(를) 대상으로 [ {i} ]의 효과를 [ {c} ]와(과) 비교하여 [ {o} ]을(를) 결과로 보고한 연구인가?"
아래 5단계의 확실성 등급 중 가장 적절한 **숫자 하나만** 반환해주세요. 다른 설명은 절대 추가하지 마세요.
5: 매우 높은 확실성, 4: 높은 확실성, 3: 중간 확실성, 2: 낮은 확실성, 1: 관련 없음 또는 판단 불가
--- 초록 시작 ---
{abstract}
--- 초록 끝 ---
확실성 등급 (숫자만):
"""

def parse_xml_records(pubmed_data):
    """
    Entrez.read()로 파싱된 PubMed XML 데이터에서 주요 정보를 추출합니다.
    이 함수는 수동 텍스트 파싱보다 훨씬 안정적입니다.

    Args:
        pubmed_data (dict): Entrez.read()로 파싱된 PubMed XML 데이터.

    Returns:
        list: 각 논문의 정보를 포함한 딕셔너리 리스트 (pmid, abstract, publication_types, mesh_terms).
    """
    records = []
    for article_data in pubmed_data.get('PubmedArticle', []):
        medline_citation = article_data.get('MedlineCitation', {})
        if not medline_citation:
            continue

        pmid = str(medline_citation.get('PMID', ''))
        article = medline_citation.get('Article', {})

        # 구조화된 초록(Abstract) 처리
        abstract_element = article.get('Abstract', {}).get('AbstractText', [])
        if isinstance(abstract_element, list):
            abstract_parts = []
            for part in abstract_element:
                label = part.attributes.get('Label', '')
                text = str(part)
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)
        else:
            abstract = str(abstract_element)

        # 출판 유형(Publication Types) 처리
        pub_type_list = article.get('PublicationTypeList', [])
        publication_types = "; ".join([str(pt) for pt in pub_type_list])

        # MeSH 용어(MeSH Terms) 처리
        mesh_heading_list = medline_citation.get('MeshHeadingList', [])
        mesh_terms_list = []
        for mesh_heading in mesh_heading_list:
            descriptor = mesh_heading.get('DescriptorName')
            if descriptor:
                mesh_terms_list.append(str(descriptor))
        mesh_terms = "; ".join(mesh_terms_list)

        records.append({
            "pmid": pmid,
            "abstract": abstract or "No Abstract Found",
            "publication_types": publication_types,
            "mesh_terms": mesh_terms
        })
    return records

def get_pico_input():
    """
    사용자로부터 PICO 정보를 입력받습니다.

    Returns:
        tuple: (population, intervention, control, outcome)
    """
    print("\n--- Systematic Review를 위한 PICO 정보 입력 ---")
    population = ""
    while not population:
        population = input("1. Population (대상 집단): ")
    intervention = ""
    while not intervention:
        intervention = input("2. Intervention (중재): ")
    control = input("3. Control (비교 대상) (선택사항, 없으면 Enter): ") or "Control 없음"
    outcome = ""
    while not outcome:
        outcome = input("4. Outcome (결과): ")
    return population, intervention, control, outcome

def analyze_abstract_with_gemini(model, abstract, p, i, c, o, pmid):
    """
    Gemini API를 사용하여 초록을 분석하고 PICO 기준에 따른 확실성 점수와 토큰 수를 반환합니다.

    Args:
        model: Gemini API 모델 객체.
        abstract (str): 분석할 논문 초록.
        p (str): Population.
        i (str): Intervention.
        c (str): Control.
        o (str): Outcome.
        pmid (str): 로깅을 위한 PubMed ID.

    Returns:
        tuple: (certainty_score, token_count)
    """
    prompt = PROMPT_TEMPLATE.format(p=p, i=i, c=c, o=o, abstract=abstract)
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        if response_text.isdigit() and 1 <= int(response_text) <= 5:
            certainty = int(response_text)
        else:
            tqdm.write(f"   - Gemini API 응답 오류: 유효하지 않은 값 '{response_text}' (PMID: {pmid})")
            return 0, 0

        tokens = model.count_tokens(prompt).total_tokens
        return certainty, tokens
    except Exception as e:
        tqdm.write(f"   - Gemini API 오류: {e} (PMID: {pmid})")
        return 0, 0

def analyze_row(row, model, p, i, c, o):
    """
    DataFrame의 각 행에 대해 초록을 분석하고 확실성 점수와 토큰 수를 반환합니다.

    Args:
        row: DataFrame의 행.
        model: Gemini API 모델 객체.
        p (str): Population.
        i (str): Intervention.
        c (str): Control.
        o (str): Outcome.

    Returns:
        tuple or None: (확실성 점수, 토큰 수) 또는 None.
    """
    abstract = row.get("abstract", "")
    pmid = row.get("pmid", "N/A")

    if pd.isna(abstract) or len(str(abstract)) < MIN_ABSTRACT_LENGTH:
        tqdm.write(f"   - 초록 내용이 짧거나 없어 건너뜁니다 (PMID: {pmid}).")
        return None

    return analyze_abstract_with_gemini(model, abstract, p, i, c, o, pmid)

def main():
    """
    메인 실행 함수
    """
    # --- 1. PubMed API 설정 및 검색어 입력 ---
    print("--- PubMed 논문 검색 및 PICO 분석 도구 ---")

    email = ""
    while not email:
        email = input("PubMed API 사용을 위해 이메일 주소를 입력해주세요: ")
    Entrez.email = email
    # Entrez.api_key = userdata.get('PUBMED_API_KEY', None) # API 키 없이 사용

    keyword = input("검색할 키워드를 입력하세요: ")
    year = input("검색할 출판 연도를 입력하세요 (예: 2023): ")
    search_term = f'({keyword}) AND ("{year}"[Date - Publication])'

    print("\nPubMed에서 논문을 검색 중입니다...")
    try:
        handle = Entrez.esearch(db="pubmed", term=search_term, usehistory="y", retmax=MAX_RETMAX)
        search_results = Entrez.read(handle)
        handle.close()
    except Exception as e:
        print(f"PubMed 검색 중 오류 발생: {e}")
        return

    count = int(search_results["Count"])
    if count == 0:
        print("검색된 논문이 없습니다.")
        return

    fetch_count = count
    if count > MAX_RETMAX:
        print(f"\n경고: 총 검색 결과 {count}개가 최대 다운로드 개수 {MAX_RETMAX}개를 초과합니다.")
        print(f"상위 {MAX_RETMAX}개의 논문만 다운로드하여 분석합니다.")
        fetch_count = MAX_RETMAX

    print(f"\n총 {count}개의 논문을 찾았습니다. 이 중 {fetch_count}개를 다운로드합니다.")

    # --- 2. EFetch로 데이터 추출 (XML 형식 사용) ---
    all_pubmed_records = []
    with tqdm(total=fetch_count, desc="데이터 다운로드 중") as pbar:
        for start in range(0, fetch_count, BATCH_SIZE):
            try:
                fetch_handle = Entrez.efetch(
                    db="pubmed", rettype="abstract", retmode="xml", retstart=start,
                    retmax=BATCH_SIZE, webenv=search_results["WebEnv"], query_key=search_results["QueryKey"]
                )
                pubmed_data = Entrez.read(fetch_handle)
                fetch_handle.close()

                parsed_records = parse_xml_records(pubmed_data)
                all_pubmed_records.extend(parsed_records)
                pbar.update(len(parsed_records))
                time.sleep(SLEEP_INTERVAL)
            except Exception as e:
                print(f"\n데이터 추출 중 오류 발생 (기록 {start+1}부터): {e}")
                continue

    df = pd.DataFrame(all_pubmed_records)
    print(f"\n총 {len(df)}개의 논문 데이터를 성공적으로 추출했습니다.")

    # --- 3. Gemini API 설정 및 PICO 입력 ---
    try:
        api_key = userdata.get('GOOGLE_API_KEY')
        if not api_key:
            print("오류: Colab의 Secrets에 'GOOGLE_API_KEY'를 설정해주세요.")
            return
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Gemini API 설정 중 오류 발생: {e}")
        return

    p, i, c, o = get_pico_input()

    # --- 4. Gemini API로 초록 분석 ---
    results = []
    total_tokens, analyzed_count = 0, 0
    print("\nGemini API를 사용하여 논문 초록 분석을 시작합니다...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="논문 분석 중"):
        analysis_result = analyze_row(row, model, p, i, c, o)

        certainty = None
        if analysis_result:
            score, tokens = analysis_result
            if score > 0:
                certainty = score
                total_tokens += tokens
                analyzed_count += 1

        results.append(certainty)
        time.sleep(1)

    # --- 5. 최종 결과를 CSV 파일로 저장 ---
    df['certainty'] = results
    output_filename = "pubmed_pico_analysis.csv"
    try:
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n모든 분석 결과가 '{output_filename}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"\nCSV 파일 저장 중 오류가 발생했습니다: {e}")

    # --- 6. 최종 통계 출력 ---
    if analyzed_count > 0:
        average_tokens = total_tokens / analyzed_count
        print("\n--- 분석 통계 ---")
        print(f"총 분석된 논문 수: {analyzed_count}개")
        print(f"사용한 총 토큰 수: {total_tokens}개")
        print(f"논문 당 평균 토큰 수: {average_tokens:.2f}개")
    else:
        print("\n분석된 논문이 없습니다.")

if __name__ == "__main__":
    main()