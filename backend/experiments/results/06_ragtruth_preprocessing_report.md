# VeriLLM Phase 7: RAGTruth Dataset Preprocessing Report

## 1. Quality Filtering and Exclusions

- Total Raw Responses: `17790`
- Good Quality: `17617` (99.0%)
- Truncated (Excluded): `29`
- Incorrect Refusals (Excluded): `144`

## 2. Partition and Class Distributions

| Partition | Sources | Total Sentences | SUPPORTED_PROXY | CONTRADICTED | UNSUPPORTED | MIXED_HALLUCINATION |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`TRAIN`** | 2140 | 89943 | 79456 (88.3%) | 3392 (3.8%) | 6646 (7.4%) | 449 (0.5%) |
| **`VAL`** | 375 | 15792 | 13970 (88.5%) | 581 (3.7%) | 1167 (7.4%) | 74 (0.5%) |
| **`TEST`** | 450 | 17943 | 16388 (91.3%) | 553 (3.1%) | 965 (5.4%) | 37 (0.2%) |
| **TOTAL** | 2965 | 123678 | 109814 (88.8%) | 4526 (3.7%) | 8778 (7.1%) | 560 (0.5%) |

## 3. Sequence Length Statistics

- **Sentence Length**: Median: `18` words, 95th: `33` words, Max: `267` words
- **Context Length**: Median: `334` words, 95th: `943` words, Max: `1751` words
- **Combined Pair Length**: Median: `352` words, 95th: `969` words, 99th: `1446` words
- **Pairs exceeding 512 words**: `24919` (20.15%)

## 4. Sample Examples Across Classes

### Class: `SUPPORTED_PROXY`
- **Example ID**: `0_s01` (Model: `gpt-4-0613`, Task: `Summary`)
- **Sentence**: *"The Anne Frank House has revealed that Anne Frank and her older sister, Margot, likely died at least a month earlier than previously believed."*

### Class: `CONTRADICTED`
- **Example ID**: `2_s04` (Model: `mistral-7B-instruct`, Task: `Summary`)
- **Sentence**: *"The sisters contracted typhus, which was rampant in the camp, and are believed to have died before February 7, 2022."*
- **Annotations**:
  - Span: *"February 7, 2022."* | Type: `Evident Conflict` | Overlap: `14.7%`
  - Span: *"believed to have died before February 7"* | Type: `Evident Conflict` | Overlap: `33.6%`

### Class: `UNSUPPORTED`
- **Example ID**: `2_s06` (Model: `mistral-7B-instruct`, Task: `Summary`)
- **Sentence**: *"The discovery has led to a reevaluation of the events surrounding Anne Frank's death and has prompted the Anne Frank House to issue a corrected statement regarding the date of her passing."*
- **Annotations**:
  - Span: *"has prompted the Anne Frank House to issue a corrected statement regarding the date of her passing"* | Type: `Evident Baseless Info` | Overlap: `52.1%`

### Class: `MIXED_HALLUCINATION`
- **Example ID**: `176_s03` (Model: `mistral-7B-instruct`, Task: `Summary`)
- **Sentence**: *"The brutal murder of Farkhunda, a young woman in Afghanistan, in 2015 shocked the world, but her feminist act of carrying her own coffin during her funeral also made international headlines."*
- **Annotations**:
  - Span: *"carrying her own coffin during her funeral"* | Type: `Evident Conflict` | Overlap: `22.1%`
  - Span: *"also made international headlines."* | Type: `Subtle Baseless Info` | Overlap: `17.9%`

