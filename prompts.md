# 테스트에 쓴 프롬프트 모음

노트북의 `PROMPTS` 딕셔너리에 그대로 붙여 쓸 수 있는 형태로 정리했습니다.

## 공통 네거티브 프롬프트

```
lowres, bad anatomy, bad hands, extra fingers, fewer fingers, missing limbs,
extra limbs, deformed, disfigured, mutated, cropped, worst quality, low quality,
jpeg artifacts, blurry, watermark, signature, text
```

## 기본 파라미터

| 항목 | 값 | 메모 |
|---|---|---|
| steps | 25 | 20 이하면 손이 깨짐 |
| guidance_scale | 7.5 | 9 넘으면 색이 타버림 |
| controlnet_conditioning_scale | 1.0 | 자세 유지 강도. 0.6이면 프롬프트 쪽으로 더 끌려감 |
| seed | 1234 | 비교용으로 고정 |

---

## 01. 같은 포즈, 프롬프트만 바꾸기 (pose_01)

참조: `samples/images1.jpg` — 정면 직립, 한 손만 바지 주머니에 넣은 자세
골격: `samples/pose_01.png`

기준:

```
a young man in a white linen shirt standing in a sunflower field,
golden hour, soft rim light, 85mm photo, shallow depth of field,
highly detailed, photorealistic
```

변형:

```
a medieval knight in polished steel armor, standing in a castle courtyard,
dramatic overcast light, cinematic, highly detailed
```

```
an astronaut in a white spacesuit standing on mars, dusty red terrain,
lens flare, cinematic lighting, highly detailed
```

```
anime illustration of a young man standing, clean lineart, cel shading,
pastel colors, studio ghibli inspired
```

## 02. 같은 프롬프트, 다른 포즈 (pose_02)

참조: `samples/images2.jpg` — 다리를 X자로 꼬고 서서 양손을 주머니에 넣은 자세
골격: `samples/pose_02.png`

01과 같은 프롬프트를 그대로 넣어, 자세만 바뀌었을 때 결과가 어떻게 달라지는지 봅니다.

```
a young man in a white linen shirt standing in a sunflower field,
golden hour, soft rim light, 85mm photo, shallow depth of field,
highly detailed, photorealistic
```

노트북의 `BASE_PROMPT`와 같은 문장입니다. 01과 02의 차이는 참조 사진뿐입니다.

## 03. 자세 유지 강도 비교용

같은 프롬프트로 `controlnet_conditioning_scale`만 바꿔 3장 뽑아 비교합니다.

```
a dancer mid-motion on a dark stage, single spotlight, dust in the air,
dramatic shadows, photorealistic, highly detailed
```

- `0.5` → 자세가 흐트러지고 프롬프트 분위기 우선
- `1.0` → 자세 거의 그대로 (기본값으로 채택)
- `1.5` → 자세는 정확하지만 그림이 뻣뻣하고 배경이 단조로워짐

---

## 프롬프트 작성 시 메모

- 자세는 ControlNet이 잡아주니 프롬프트에 자세 설명을 길게 쓸 필요는 없다. `standing` 한 단어 정도만 넣어주면 충분하다.
- 인물 수는 참조 사진의 골격 개수가 결정한다. 프롬프트에 `two people`이라 써도 골격이 하나면 한 명만 나온다.
- 손이 깨질 때는 프롬프트보다 `steps`를 올리는 게 효과가 크다 (25 → 35).
- 주머니에 손을 넣은 자세는 손 키포인트가 몸에 가려 일부만 잡힌다. 그 부분은 프롬프트로 보완해야 한다 (`hands in pockets`).
- 얼굴 방향이 어긋날 때는 OpenPose 추출 시 `include_face=True`가 켜져 있는지 확인.
