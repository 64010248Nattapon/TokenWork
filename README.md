# GreenMind — สูตร 1 พร้อมส่ง

Weight ที่ใช้ส่งมีไฟล์เดียวบน Hugging Face ไม่ได้แนบน้ำหนักในโฟลเดอร์นี้

**https://huggingface.co/Nattapon248/mango15-sq-gptq-w8a8**

Revision ที่ log เปิดจริง: `1ce063a2d48c1b4cfcc7981730d0429297e5af58`

เป็น SmoothQuant α 0.8 แล้ว GPTQ แบบ W8A8 (compressed-tensors) จาก `quant/Quantize_StepByStep.ipynb` สูตร 1 `lm_head`, vision, MTP และ `linear_attn.in_proj_a/b` คงเป็น BF16

## เปิดให้ grader

ต้องมี vLLM 0.27.1 ใน environment และตั้ง `VLLM_USE_FLASHINFER_SAMPLER=0` (เครื่องนี้ไม่มี `/usr/local/cuda` สคริปต์ตั้งไว้ใน `serving/env.sh` แล้ว)

```bash
cd folder_TokenWork
FINAL_MODEL=Nattapon248/mango15-sq-gptq-w8a8 \
SERVED_NAME=shogun \
bash serving/final.sh
```

`serving/final.sh` ตั้งค่าเดียวกับรอบที่ได้ 4.47×: `THINK_BUDGET=2048`, prefix caching, SSM state เป็น bfloat16, `max_num_batched_tokens=8192`, long-prefill 2048, context 49152, GPU memory utilization 0.90, reasoning parser `qwen3`, tool parser `qwen3_coder`, ปิด MTP

หลักฐานการสตาร์ตอยู่ที่ `results/serve-nutmodel.log` บรรทัด 1, 5, 8, 14–17, 27 และ 38

## สิ่งที่อยู่ในโฟลเดอร์

| ที่ | คือ |
|---|---|
| `serving/final.sh` `env.sh` `candidate.sh` `vllm_w4a16.sh` `vllm_launch.py` | คำสั่งและ config ที่ส่ง |
| `serving/` ที่เหลือ | baseline, MTP, FP8, SGLang, GPU tune/lock ที่ลองแล้วไม่ใช้เป็นชุดส่ง |
| `quant/Quantize_StepByStep.ipynb` | โค้ดที่สร้าง weight สูตร 1 |
| `quant/w8a8_gptq.py` `w4a16_awq.py` `w8a8_int8.py` `splice_mtp.py` | สูตรอื่นที่วัดเทียบ |
| `results/serve-nutmodel.log` | log เปิด weight นี้ด้วย flag ชุดสุดท้าย |
| `slides/deck.md` | ต้นฉบับสไลด์ ไฟล์ PDF ยังต้อง export แยกตามข้อกำหนด |

## สิ่งที่ลองแล้วไม่ได้ใช้เป็นชุดส่ง

- Self-calibration สร้าง trace ไม่สำเร็จ จึง calibrate ด้วยข้อมูลสาธารณะ 384 ตัวอย่าง และไม่อ้างผลจาก thinking trace
- MTP พร้อมการเปลี่ยน scheduler ได้ S 86.1 และ 240 J/request จึงไม่เลือก ชุดที่ส่งปิด speculative decoding
- รอบที่มี traffic อื่นซ้อนอยู่ ไม่ใช้สรุปความเร็วหรือผลของการล็อกความถี่
- W4A16, mixed W4/W8 และ W8A16 ใช้ดูการแลกคุณภาพกับขนาด ชุดที่ส่งคือ W8A8 เพราะมี INT8 kernel จริงและคุณภาพผ่านเกณฑ์ใน Final
