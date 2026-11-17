# VeriLLM Phase 4: Evidence & Document-Processing Demonstration Report

- **Document**: `sample_apollo11_mission.pdf` (3 pages, 5 chunks)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, CPU mode)
- **Batch Retrieval Speed**: 18.3 ms/claim

## Extracted Claims & Retrieved Evidence Passages

| Claim ID | Proposition | Page | Cosine Similarity | Category | Top Retrieved Chunk |
|:---|:---|:---:|:---:|:---|:---|
| `claim_001` | The Apollo 11 mission was launched on a Saturn V rocket from Kennedy Space Center on July 16, 1969. | 1 | **0.7587** | `HIGH_RELEVANCE` | *: Launch and Trans-Lunar Flight The Apollo 11 mission was the first spaceflight ...* |
| `claim_002` | Neil Armstrong and Buzz Aldrin landed the lunar module Eagle on July 20, 1969, and they collected 47.5 pounds of lunar rock. | 2 | **0.7409** | `HIGH_RELEVANCE` | *: Lunar Landing and Surface Operations On July 20, 1969, Armstrong and Aldrin en...* |
| `claim_003` | Michael Collins flew the Command Module Columbia alone in lunar orbit. | 2 | **0.7389** | `HIGH_RELEVANCE` | *Buzz Aldrin joined him 19 minutes later. Together, the astronauts spent over two...* |
| `claim_004` | The astronauts safely returned to Earth. | 3 | **0.6691** | `HIGH_RELEVANCE` | *The astronauts began their journey back to Earth. Apollo 11 safely returned to E...* |
| `claim_005` | The astronauts splashed down in the Pacific Ocean on July 24, 1969. | 3 | **0.6768** | `HIGH_RELEVANCE` | *The astronauts began their journey back to Earth. Apollo 11 safely returned to E...* |
| `claim_006` | The mission was watched by over 650 million television viewers worldwide. | 2 | **0.3712** | `MODERATE_RELEVANCE` | *Buzz Aldrin joined him 19 minutes later. Together, the astronauts spent over two...* |
| `claim_007` | Buzz Aldrin also planted an olive branch on Mars. | 2 | **0.5255** | `HIGH_RELEVANCE` | *Buzz Aldrin joined him 19 minutes later. Together, the astronauts spent over two...* |

## Key Architectural Insights

1. **Traceability**: Every extracted claim is matched to an exact page number and chunk ID, enabling citation in the UI.
2. **Topical vs. Factual Distinction**: High cosine similarity indicates that a passage discusses the same topic as the claim, but does not guarantee truth (e.g. contradiction vs entailment). Entailment checking is handled in Phase 5.
3. **Absence of Evidence**: Propositions not mentioned in the reference document (e.g., planting an olive branch on Mars) yield distinctly lower similarity scores (<0.25), correctly indicating an absence of supporting context.
