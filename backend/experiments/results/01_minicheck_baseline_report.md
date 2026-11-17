# VeriLLM Phase 2: MiniCheck Baseline Experiment Report

- **Date / Time**: 2026-11-08T18:05:28
- **Model**: `roberta-large` (`lytang/MiniCheck-RoBERTa-Large`, 355M parameters)
- **Runtime Environment**: CPU-only PyTorch (Intel UHD 620 laptop environment)
- **Model Load Time**: 4.76 seconds

---

## 1. Controlled Reference Document
> "Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC. Neil Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later, on July 21 at 02:56 UTC. Aldrin joined him 19 minutes later. They spent about two and a quarter hours together outside the spacecraft, and collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth. Command Module Pilot Michael Collins flew the Command Module Columbia alone in lunar orbit while they were on the surface."

---

## 2. Experimental Results

| Test Case | Target Category | Expected Label | Predicted Label | Support Probability | Inference Latency | Validation |
|---|---|---|---|---|---|---|
| `claim_01_supported` | **SUPPORTED** | `1` | `1` | **0.9891** | 1.530 s | **PASS** |
| `claim_02_contradicted` | **CONTRADICTED** | `0` | `0` | **0.0392** | 1.290 s | **PASS** |
| `claim_03_unmentioned` | **UNMENTIONED** | `0` | `0` | **0.0037** | 1.494 s | **PASS** |

---

## 3. Individual Claim Analysis

### `claim_01_supported` (SUPPORTED)
- **Claim**: "Neil Armstrong was the first person to step onto the lunar surface on July 21, 1969."
- **Description**: Directly stated in the document (July 21, 02:56 UTC).
- **Predicted Output**: `label = 1`, `support_prob = 0.9891`
- **Outcome**: Correctly classified as expected.

### `claim_02_contradicted` (CONTRADICTED)
- **Claim**: "Buzz Aldrin flew the Command Module Columbia alone in lunar orbit."
- **Description**: Contradicts document: Michael Collins flew Columbia alone, while Aldrin was on the surface.
- **Predicted Output**: `label = 0`, `support_prob = 0.0392`
- **Outcome**: Correctly classified as expected.

### `claim_03_unmentioned` (UNMENTIONED)
- **Claim**: "The Apollo 11 astronauts ate bacon and freeze-dried peaches for breakfast on their return flight."
- **Description**: Not mentioned anywhere in the reference document (outside knowledge / ungrounded).
- **Predicted Output**: `label = 0`, `support_prob = 0.0037`
- **Outcome**: Correctly classified as expected.

---

## 4. Scientific Observations & Known Baseline Limitations

1. **Binary Classification Ambiguity**:
   MiniCheck evaluates claims on a single binary dimension: `1` (supported by the document) vs `0` (unsupported by the document).
   - Case 2 (*Contradicted*: Buzz Aldrin flew Columbia alone) receives `label = 0` with low support probability.
   - Case 3 (*Unmentioned*: Astronauts ate bacon and peaches) also receives `label = 0` with low support probability.
   - **Research Implication for VeriLLM**: A user cannot determine from MiniCheck alone whether an unsupported claim directly contradicts the facts or is simply unmentioned. VeriLLM's Phase 5 and Phase 7 directly solve this by implementing a 4-way classification pipeline (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`, `UNVERIFIABLE`).

2. **Document Grounding vs. Universal Truth**:
   MiniCheck strictly measures document faithfulness. If an LLM generates a claim that is true in the real world but absent from the reference text, MiniCheck will classify it as `unsupported`. This is intentional and necessary for grounded evaluation (e.g. RAG pipelines).
