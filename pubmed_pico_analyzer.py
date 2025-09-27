import time
import csv
import pandas as pd
import google.generativeai as genai
from google.colab import userdata
from Bio import Entrez
from tqdm import tqdm

# 상수 정의
BATCH_SIZE = 200
SLEEP_INTERVAL = 0.4 # PubMed API 호출 사이의 간격 (초)
MAX_RETMAX = 10000 # PubMed에서 한 번에 검색할 최대 논문 수
MIN_ABSTRACT_LENGTH = 50 # 분석을 위한 최소 초록 길이

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

def parse_medline_records(medline_text):
    """
    MEDLINE 텍스트를 파싱하여 PMID, 초록, PT, MeSH 정보를 추출합니다.

    Args:
        medline_text (str): PubMed에서 가져온 MEDLINE 형식의 텍스트 데이터.

    Returns:
        list: 각 논문의 정보를 포함한 딕셔너리 리스트 (pmid, abstract, publication_types, mesh_terms).
    """
    records = []
    # PMID- 를 기준으로 개별 논문 데이터를 분리합니다.
    individual_articles = medline_text.strip().split("PMID- ")

    for article_text in individual_articles:
        if not article_text.strip():
            continue

        pmid, abstract = "", ""
        pub_types, mesh_terms = [], []

        lines = article_text.strip().split("\n")
        if not lines:
            continue

        pmid = lines[0].strip()

        is_abstract_section = False
        abstract_lines = []

        for line in lines:
            # 'AB  - '로 시작하는 줄은 초록의 시작입니다.
            if line.startswith("AB  - "):
                is_abstract_section = True
                abstract_lines.append(line[6:].strip())
            # 초록 섹션이 계속되는 경우 (들여쓰기 된 줄)
            elif is_abstract_section and line.startswith("      "):
                abstract_lines.append(line[6:].strip())
            else:
                is_abstract_section = False

            if line.startswith("PT  - "):
                pub_types.append(line[6:].strip())

            if line.startswith("MH  - "):
                mesh_terms.append(line[6:].strip())

        if abstract_lines:
            abstract = " ".join(abstract_lines)

        records.append({
            "pmid": pmid,
            "abstract": abstract or "No Abstract Found",
            "publication_types": "; ".join(pub_types),
            "mesh_terms": "; ".join(mesh_terms)
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

def analyze_abstract_with_gemini(model, abstract, p, i, c, o):
    """
    Gemini API를 사용하여 초록을 분석하고 PICO 기준에 따른 확실성 점수와 토큰 수를 반환합니다.

    Args:
        model: Gemini API 모델 객체.
        abstract (str): 분석할 논문 초록.
        p (str): Population.
        i (str): Intervention.
        c (str): Control.
        o (str): Outcome.

    Returns:
        tuple: (certainty_score, token_count)
    """
    prompt = PROMPT_TEMPLATE.format(p=p, i=i, c=c, o=o, abstract=abstract)
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # 응답이 숫자인지, 그리고 1-5 범위 내에 있는지 확인합니다.
        if response_text.isdigit() and 1 <= int(response_text) <= 5:
            certainty = int(response_text)
        else:
            # 유효하지 않은 응답일 경우, 로그를 남기고 0을 반환합니다.
            tqdm.write(f"   - Gemini API 응답 오류: 유효하지 않은 값 '{response_text}'")
            return 0, 0

        tokens = model.count_tokens(prompt).total_tokens
        return certainty, tokens
    except Exception as e:
        tqdm.write(f"   - Gemini API 오류: {e}")
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
    # 초록이 비어있거나 너무 짧으면 분석을 건너뜁니다.
    # "No Abstract Found" 또는 의미 없는 짧은 문자열을 걸러내기 위함입니다.
    if pd.isna(abstract) or len(str(abstract)) < MIN_ABSTRACT_LENGTH:
        tqdm.write(f"   - 초록 내용이 짧거나 없어 건너뜁니다 (PMID: {row['pmid']}).")
        return None

    return analyze_abstract_with_gemini(model, abstract, p, i, c, o)

def main():
    """
    메인 실행 함수
    """
    # --- 1. PubMed API 설정 및 검색어 입력 ---
    print("--- PubMed 논문 검색 및 PICO 분석 도구 ---")

    # 사용자에게 이메일 주소 입력을 요청 (PubMed API 정책)
    email = ""
    while not email:
        email = input("PubMed API 사용을 위해 이메일 주소를 입력해주세요: ")
    Entrez.email = email
    Entrez.api_key = userdata.get('PUBMED_API_KEY', None)

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

    print(f"총 {count}개의 논문을 찾았습니다. MEDLINE 데이터를 다운로드합니다.")

    # --- 2. EFetch로 데이터 추출 ---
    all_pubmed_records = []
    with tqdm(total=count, desc="데이터 다운로드 중") as pbar:
        for start in range(0, count, BATCH_SIZE):
            try:
                fetch_handle = Entrez.efetch(
                    db="pubmed", rettype="medline", retmode="text", retstart=start,
                    retmax=BATCH_SIZE, webenv=search_results["WebEnv"], query_key=search_results["QueryKey"]
                )
                medline_data = fetch_handle.read()
                fetch_handle.close()
                parsed_records = parse_medline_records(medline_data)
                all_pubmed_records.extend(parsed_records)
                pbar.update(len(parsed_records))
                time.sleep(SLEEP_INTERVAL)
            except Exception as e:
                print(f"\n데이터 추출 중 오류 발생 (PMID {start+1}부터): {e}")
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
    certainty_scores = []
    total_tokens, analyzed_count = 0, 0
    print("\nGemini API를 사용하여 논문 초록 분석을 시작합니다...")

    # tqdm을 사용하여 진행 상황 표시
    for _, row in tqdm(df.iterrows(), total=len(df), desc="논문 분석 중"):
        analysis_result = analyze_row(row, model, p, i, c, o)

        if analysis_result:
            certainty, tokens = analysis_result
            if certainty > 0:
                certainty_scores.append(certainty)
                total_tokens += tokens
                analyzed_count += 1
            else:
                certainty_scores.append(None) # API 오류 또는 유효하지 않은 응답
        else:
            certainty_scores.append(None) # 분석 건너뜀

        # Gemini API의 분당 요청 제한(기본 60 QPM)을 준수하기 위해 1초 대기
        time.sleep(1)

    # --- 5. 최종 결과를 CSV 파일로 저장 ---
    df['certainty'] = certainty_scores
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