# =============================================================================
#  pose-image-tool — Colab 붙여넣기용
# =============================================================================
#  아래 세 덩어리를 Colab 셀 3개에 각각 붙여 넣고 위에서부터 실행하세요.
#  파일 업로드 창이 뜨지 않습니다. 참조 사진을 GitHub에서 바로 받아옵니다.
#
#  실행 전: 런타임 → 런타임 유형 변경 → T4 GPU
#  셀 3만 고쳐서 몇 번이고 다시 실행하면 됩니다 (모델은 이미 올라가 있음).
# =============================================================================


# =============================================================================
#  [셀 1] 설치 — 2~3분. 한 번만 실행하면 됩니다.
# =============================================================================
# 설치 중 나오는 아래 메시지는 무시해도 됩니다.
#   - pip's dependency resolver ... / timm 0.6.7 다운그레이드
#     controlnet_aux가 timm<=0.6.7을 요구해서 생기는 것으로, OpenPose에는 영향 없음
#   - The module 'mediapipe' is not installed
#     controlnet_aux의 다른 검출기용 경고. OpenPose는 쓰지 않음
# 재시작 팝업이 떠도 재시작하지 않아도 됩니다.

%pip install -q "diffusers>=0.31" "transformers>=4.44" "accelerate>=0.34" \
                "controlnet_aux==0.0.9" "safetensors" "sentencepiece" "sacremoses"
print("설치 완료")


# =============================================================================
#  [셀 2] 준비 — 참조 사진 받기 → 골격 추출 → 모델 로드 → 번역기 준비
#          처음 실행 시 모델 약 4.3GB를 내려받습니다 (3~5분).
# =============================================================================
import io
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image

TARGET = 512   # 생성 해상도. 768로 올리면 품질↑ VRAM↑ (무료 T4는 512 권장)
REPO = "https://raw.githubusercontent.com/Benji5526/pose-image-tool/main/samples"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch:", torch.__version__)
if DEVICE == "cpu":
    print("경고: GPU가 없어 이미지 한 장에 10분 이상 걸립니다.")


# ----- 이미지 불러오기 -------------------------------------------------------
def fit(img, target=TARGET):
    """긴 변을 target에 맞추고 가로·세로를 8의 배수로 정리한다 (SD 요구사항)."""
    img = img.convert("RGB")
    w, h = img.size
    s = target / max(w, h)
    w, h = int(w * s), int(h * s)
    return img.resize((w - w % 8, h - h % 8), Image.LANCZOS)


def load_url(url):
    """이미지 URL에서 바로 불러온다. 업로드 창이 필요 없다."""
    r = requests.get(url, timeout=60, headers={"User-Agent": "pose-image-tool"})
    r.raise_for_status()
    return fit(Image.open(io.BytesIO(r.content)))


def load_upload():
    """내 사진을 쓰고 싶을 때만 호출. 파일 선택창이 뜬다."""
    from google.colab import files
    up = files.upload()
    name = next(iter(up))
    print("업로드:", name)
    return fit(Image.open(io.BytesIO(up[name])))


# ----- 포즈 추출 -------------------------------------------------------------
from controlnet_aux import OpenposeDetector

openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")


def extract_pose(img):
    """사진에서 골격 이미지를 뽑아 원본과 같은 크기로 돌려준다."""
    pose = openpose(
        img,
        include_body=True, include_hand=True, include_face=True,
        detect_resolution=512, image_resolution=max(img.size),
    )
    pose = pose.resize(img.size, Image.LANCZOS)
    ratio = float((np.asarray(pose).max(axis=2) > 24).mean())
    print(f"  골격 픽셀 {ratio:.2%} -> {'검출 성공' if ratio > 0.01 else '검출 실패: 다른 사진을 쓰세요'}")
    return pose


print("\n참조 사진 받는 중...")
ref_1 = load_url(f"{REPO}/images1.jpg")     # 정면 직립, 한 손 주머니
ref_2 = load_url(f"{REPO}/images2.jpg")     # 다리 X자로 꼬고 양손 주머니
pose_1 = extract_pose(ref_1)
pose_2 = extract_pose(ref_2)

fig, ax = plt.subplots(1, 4, figsize=(13, 4))
for a, img, t in zip(ax, [ref_1, pose_1, ref_2, pose_2],
                     ["ref 1", "pose 1", "ref 2", "pose 2"]):
    a.imshow(img); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()


# ----- 생성 파이프라인 -------------------------------------------------------
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=DTYPE)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    # runwayml/stable-diffusion-v1-5는 이전되어 지금은 리다이렉트로만 접근됩니다.
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, torch_dtype=DTYPE,
    safety_checker=None, requires_safety_checker=False,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

if DEVICE == "cuda":
    pipe.enable_attention_slicing()
    # 주의: 이 호출이 모델을 알아서 GPU로 옮깁니다.
    #      여기서 pipe.to("cuda")를 또 부르면 diffusers가 오류를 냅니다.
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(DEVICE)

NEGATIVE = (
    "lowres, bad anatomy, bad hands, extra fingers, fewer fingers, missing limbs, "
    "extra limbs, deformed, disfigured, mutated, cropped, worst quality, low quality, "
    "jpeg artifacts, blurry, watermark, signature, text"
)


def generate(prompt, pose, negative=NEGATIVE, steps=25,
             guidance=7.5, pose_scale=1.0, seed=1234):
    """골격 + 프롬프트 -> 이미지 1장."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return pipe(
        prompt=prompt, negative_prompt=negative, image=pose,
        num_inference_steps=steps, guidance_scale=guidance,
        controlnet_conditioning_scale=float(pose_scale), generator=gen,
    ).images[0]


# ----- 한국어 프롬프트 번역 --------------------------------------------------
_HAN = re.compile(r"[가-힣]")
_MT = {}       # 번역 모델은 처음 쓸 때만 올린다 (약 300MB)

# 번역기가 자주 흘리는 낱말. 원문에 있는데 번역문에 없으면 영어 표현을 덧붙인다.
HINTS = {
    "남자": "a man", "여자": "a woman", "청년": "a young man",
    "소년": "a boy", "소녀": "a girl", "노인": "an elderly person",
    "기사": "a knight", "우주인": "an astronaut",
    "발레리나": "a ballerina", "무용수": "a dancer",
    "황금빛": "golden hour", "노을": "golden hour", "역광": "backlit",
    "흐린": "overcast light", "맑은": "clear sky", "실내": "indoor light",
    "극적": "dramatic lighting", "스포트라이트": "single spotlight",
    "영화": "cinematic", "사진": "photo", "실사": "photorealistic",
    "고화질": "highly detailed", "유화": "oil painting", "수채": "watercolor",
    "애니": "anime illustration", "만화": "comic illustration",
    "흑백": "black and white",
}


def to_en(text):
    """한국어가 섞여 있으면 영어로 번역한다. 영어만 있으면 그대로 돌려준다."""
    if not _HAN.search(text):
        return text
    if "tok" not in _MT:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = "Helsinki-NLP/opus-mt-ko-en"
        print("번역 모델 로드 중 (처음 한 번만, 약 300MB)...")
        _MT["tok"] = AutoTokenizer.from_pretrained(name)
        _MT["mdl"] = AutoModelForSeq2SeqLM.from_pretrained(name)
    tok, mdl = _MT["tok"], _MT["mdl"]

    batch = tok([text], return_tensors="pt", padding=True)
    gen = mdl.generate(**batch, num_beams=4, max_new_tokens=160)
    out = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

    # 관사를 떼고 비교해야 "young man"이 있는데 "a young man"을 또 붙이지 않는다.
    low, extra = out.lower(), []
    for ko, en in HINTS.items():
        core = en.lower()
        for art in ("a ", "an ", "the "):
            if core.startswith(art):
                core = core[len(art):]
                break
        if ko in text and core not in low and en not in extra:
            extra.append(en)
    return (out.rstrip(" .") + ", " + ", ".join(extra)) if extra else out


print("\n준비 완료. 셀 3으로 넘어가세요.")


# =============================================================================
#  [셀 3] 작업 칸 — 여기만 고쳐서 반복 실행하세요
# =============================================================================
#  주의: 자세는 프롬프트가 아니라 MY_POSE의 골격이 결정합니다.
#        "점프하는" 이라고 써도 골격이 서 있으면 서 있는 그림이 나옵니다.
#        다른 자세를 원하면 그 자세의 참조 사진이 필요합니다:
#          ref_3 = load_upload()          # 내 사진 올리기
#          ref_3 = load_url("https://...")  # 인터넷 이미지
#          pose_3 = extract_pose(ref_3)
#          MY_POSE = pose_3

MY_PROMPTS_KO = {
    # "이름": "한국어로 써도 되고 영어로 써도 됩니다"
    # 이름은 그래프 제목으로 쓰이니 영문으로 적으세요 (한글은 □로 깨집니다).
    "try1": "해바라기 밭에 서 있는 흰 셔츠를 입은 남자, 황금빛 노을, 사진, 고화질",
    "try2": "갑옷을 입은 중세 기사, 성 안뜰, 흐린 날, 영화 같은 조명",
}

MY_POSE = pose_1                    # pose_1(직립) 또는 pose_2(다리 꼬기)
MY_PARAMS = dict(steps=25,          # 손이 깨지면 35
                 guidance=7.5,      # 프롬프트 충실도
                 pose_scale=1.0,    # 자세 유지 강도 0.5~1.5
                 seed=1234)         # 비교할 때는 고정

# --- 여기서부터는 고치지 않아도 됩니다 ---------------------------------------
MY_PROMPTS = {k: to_en(v) for k, v in MY_PROMPTS_KO.items()}
print("번역 결과 — 어색하면 위 한국어를 영어로 직접 고쳐 쓰세요")
for k in MY_PROMPTS:
    print(f"  [{k}] {MY_PROMPTS_KO[k]}")
    print(f"       -> {MY_PROMPTS[k]}")

results = {}
for k, p in MY_PROMPTS.items():
    print(f"생성 중: {k}")
    results[k] = generate(p, MY_POSE, **MY_PARAMS)

fig, ax = plt.subplots(1, len(results) + 1, figsize=(3.2 * (len(results) + 1), 4.2))
ax[0].imshow(MY_POSE); ax[0].set_title("pose"); ax[0].axis("off")
for a, (k, img) in zip(ax[1:], results.items()):
    a.imshow(img); a.set_title(k); a.axis("off")
plt.tight_layout(); plt.show()

# 저장 + 내 컴퓨터로 내려받기
os.makedirs("out", exist_ok=True)
for k, img in results.items():
    img.save(f"out/{k}.png")
MY_POSE.save("out/pose_used.png")
print("저장:", sorted(os.listdir("out")))

# 내려받기가 필요하면 아래 두 줄의 주석을 푸세요.
# from google.colab import files
# for k in results: files.download(f"out/{k}.png")
