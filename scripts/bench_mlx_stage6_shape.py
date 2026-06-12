"""One-shot benchmark: MLXLLMAdapter shared-prefix batch on stage-6-shape
prompts.

Question: does the 1.71x speedup measured on stage-4-shape workloads
(~500 tok prefix, ~50 tok suffix, N=3) extrapolate to the projected
3-30x range on stage-6-shape workloads (~1500 tok prefix, ~50 tok
suffix, N=3)?

Stage-6 chest-pain encounters have substantially longer prefixes
because the encounter context (chief complaint + vitals + H&P + labs)
plus prior CoT accumulates over the trajectory. The shared-prefix
saving compounds with prefix length, so longer prefixes should give
larger speedups — IF the prefill is dominated by the shared portion
(which it is, by construction in stage-6 trajectories).

This benchmark runs once and writes results to
docs/exploration/2026-05-07-mlx-stage-6-shape-speedup.md.
"""
from __future__ import annotations

import time

from bsig.reference.llm_mlx import MLXLLMAdapter

# Synthetic stage-6-shape prefix — modeled on
# clinical/prompts/cot_continuation_progressive.txt with realistic
# clinical narrative density. The point is to reach ~1500 tokens of
# shared prefix, not to be clinically authoritative.
_PREFIX = """You are an emergency-medicine resident evaluating a patient who has just presented to the ED with chest pain. The patient is a 58-year-old man who walked into triage thirty minutes ago accompanied by his wife. He describes substernal chest pressure that began approximately two hours prior to arrival while he was watching television at home. The pain radiates to his left arm and jaw, is described as "heavy" and "squeezing", and is rated 7/10 in severity. He denies dyspnea at rest but reports mild diaphoresis at onset which has since resolved. He has no associated nausea or vomiting. He has no prior episodes of similar pain. His past medical history is significant for hypertension diagnosed eight years ago (controlled on lisinopril 20 mg daily), hyperlipidemia (atorvastatin 40 mg daily), and a 35 pack-year smoking history with successful cessation four years ago. There is a strong family history of premature coronary artery disease — his father had a fatal myocardial infarction at age 52 and an older brother underwent coronary artery bypass grafting at age 56. He denies recreational drug use and consumes alcohol occasionally on weekends. His vital signs at triage were: blood pressure 158/94 mmHg, heart rate 88 beats per minute, respiratory rate 18 per minute, oxygen saturation 97% on room air, temperature 98.4 degrees Fahrenheit. On focused examination he appears mildly anxious but in no acute distress. Cardiovascular examination reveals a regular rhythm with no murmurs, rubs, or gallops; jugular venous pressure is not elevated; peripheral pulses are equal bilaterally. Pulmonary examination demonstrates clear breath sounds bilaterally without wheeze, crackles, or rhonchi. Abdominal examination is benign with no tenderness, masses, or organomegaly. Neurological examination is grossly non-focal. The initial 12-lead electrocardiogram performed within five minutes of arrival shows normal sinus rhythm at 84 beats per minute, normal axis, no ST-segment elevation or depression, no T-wave inversion, no Q waves, and no conduction abnormalities — interpreted as normal for age. Initial laboratory studies have been sent and are pending. The patient has been placed on telemetry monitoring, given 324 mg of aspirin chewed, and a peripheral intravenous line has been established. A second 12-lead electrocardiogram is planned for thirty minutes after the first.

Prior reasoning at the chief-complaint timestep (t0):

Reasoning step 1: The patient's presentation is classic for acute coronary syndrome. Substernal chest pressure with radiation to the left arm and jaw, heavy/squeezing quality, associated diaphoresis, and 7/10 severity are all hallmark features of cardiac ischemia. The two-hour duration places this within the window where intervention substantially affects outcomes if STEMI is identified.

Reasoning step 2: Demographic factors strongly elevate cardiac risk. A 58-year-old man with hypertension, hyperlipidemia, prior heavy smoking history, and a strong family history of premature CAD has substantial pre-test probability for CAD. The HEART score components from history alone (typical pain pattern, age, multiple risk factors) place this patient in the moderate-to-high risk category before any objective testing.

Reasoning step 3: The differential at this stage must include: A) acute coronary syndrome (STEMI, NSTEMI, unstable angina) — highest priority given presentation; B) aortic dissection — must be actively excluded given hypertension, although the pain quality argues against it; C) pulmonary embolism — pre-test probability lower without dyspnea or risk factors but cannot be excluded without further testing; D) musculoskeletal chest pain — possible but would not explain the diaphoresis; E) gastroesophageal causes — possible but the radiation pattern and risk factors argue for cardiac etiology first.

New evidence at the post-ECG timestep (t1):

The first 12-lead ECG shows normal sinus rhythm with no acute ischemic changes. This is informative but not exclusionary — early in the course of acute coronary syndrome the ECG is frequently non-diagnostic, and serial ECGs are required. The absence of ST elevation rules out STEMI in this snapshot but does not exclude NSTEMI or unstable angina, both of which can present with normal initial ECG. The first troponin assay is pending and will be the key discriminator over the next several hours. Aspirin has been administered.

Continue your reasoning, integrating the new evidence at the post-ECG timestep. Update your working differential — diagnoses the new evidence makes MORE likely should rise; diagnoses it RULES OUT should be removed; the leading diagnosis may change.

Reasoning step 4:"""

_COT_DIVERGENT_SUFFIX_A = " The normal initial ECG reduces the probability of STEMI as the explanation for the current pain but does not exclude NSTEMI or unstable angina. The pre-test probability for ACS remains high given the clinical presentation and risk factor profile.\n\nReasoning step 5: The next discriminator will be the troponin assay. A negative high-sensitivity troponin at presentation plus a negative repeat at three hours has a high negative predictive value for ruling out NSTEMI. Until those return, the working assumption must remain that ACS is the leading diagnosis.\n\nUpdated working differential:\nA: NSTEMI / unstable angina\nB: Aortic dissection\nC: Pulmonary embolism\nD: Other (musculoskeletal / GI)\n\nMost likely diagnosis at this timestep is\nThe best answer is"

_COT_DIVERGENT_SUFFIX_B = " The unchanged ECG combined with the patient remaining hemodynamically stable supports continuing the ACS workup along the unstable-angina or NSTEMI pathway rather than the STEMI pathway.\n\nReasoning step 5: Aortic dissection remains on the differential and the troponin assay alone will not discriminate. A chest CT angiogram should be considered if any features of dissection emerge — pulse asymmetry, aortic insufficiency, or worsening pain quality.\n\nUpdated working differential:\nA: NSTEMI / unstable angina\nB: Aortic dissection\nC: Pulmonary embolism\nD: Other (musculoskeletal / GI)\n\nMost likely diagnosis at this timestep is\nThe best answer is"

_COT_DIVERGENT_SUFFIX_C = " The clinical-decision pathway now hinges on the troponin trajectory. A negative initial troponin would not exclude ACS at two hours from pain onset but would lower the immediate-MI probability; a positive initial troponin would essentially confirm NSTEMI given the presentation. In parallel the patient should be reassessed for any features that would prompt expedited imaging for aortic pathology.\n\nReasoning step 5: The disposition decision will follow troponin and serial ECG results over the next three to six hours.\n\nUpdated working differential:\nA: NSTEMI / unstable angina\nB: Aortic dissection\nC: Pulmonary embolism\nD: Other (musculoskeletal / GI)\n\nMost likely diagnosis at this timestep is\nThe best answer is"

_TOKEN_SET = ("A", "B", "C", "D")


def _count_tokens(adapter: MLXLLMAdapter, text: str) -> int:
    _, tokenizer = adapter._ensure_loaded()
    return len(tokenizer.encode(text))


def main() -> None:
    print("Loading MLXLLMAdapter...")
    adapter = MLXLLMAdapter()
    adapter._ensure_loaded()

    prompts = [
        _PREFIX + _COT_DIVERGENT_SUFFIX_A,
        _PREFIX + _COT_DIVERGENT_SUFFIX_B,
        _PREFIX + _COT_DIVERGENT_SUFFIX_C,
    ]
    n_prompts = len(prompts)
    prefix_len = _count_tokens(adapter, _PREFIX)
    full_lens = [_count_tokens(adapter, p) for p in prompts]
    suffix_lens = [fl - prefix_len for fl in full_lens]
    print(
        f"Stage-6-shape prompts: prefix={prefix_len} tokens, "
        f"suffixes={suffix_lens}, N={n_prompts}"
    )

    # Warm-up to prime kernels (avoid first-call overhead skewing the
    # baseline)
    print("Warm-up...")
    _ = adapter.get_token_probabilities(prompts[0], _TOKEN_SET)

    # Sequential baseline (no shared-prefix path)
    print("Sequential baseline...")
    t0 = time.time()
    seq_results = [
        adapter.get_token_probabilities(p, _TOKEN_SET) for p in prompts
    ]
    seq_wall = time.time() - t0
    print(f"  wall: {seq_wall:.2f}s ({seq_wall / n_prompts:.2f}s per prompt)")

    # Shared-prefix batch
    print("Shared-prefix batch...")
    t0 = time.time()
    batch_results = adapter.get_token_probabilities_batch(prompts, _TOKEN_SET)
    batch_wall = time.time() - t0
    print(f"  wall: {batch_wall:.2f}s ({batch_wall / n_prompts:.2f}s per prompt)")

    speedup = seq_wall / batch_wall if batch_wall > 0 else float("inf")
    print(f"\nSpeedup: {speedup:.2f}x")

    # Sanity check: argmaxes consistent
    print("\nArgmax consistency:")
    for i, (sr, br) in enumerate(zip(seq_results, batch_results, strict=True)):
        s_argmax = max(sr.distribution, key=lambda k: sr.distribution[k])
        b_argmax = max(br.distribution, key=lambda k: br.distribution[k])
        agree = "✓" if s_argmax == b_argmax else "✗"
        print(
            f"  prompt {i}: seq={s_argmax} ({sr.distribution[s_argmax]:.3f}) "
            f"batch={b_argmax} ({br.distribution[b_argmax]:.3f}) {agree}"
        )
        print(
            f"    seq mass_capture={sr.mass_capture:.4f} "
            f"batch mass_capture={br.mass_capture:.4f} "
            f"|Δ|={abs(sr.mass_capture - br.mass_capture):.4f}"
        )


if __name__ == "__main__":
    main()
