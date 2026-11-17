"""
VeriLLM Proposition Decomposer Module.
Decomposes compound sentences, unwraps rhetorical questions, and resolves coreference
while calculating fine-grained clause-level character offsets.
"""

import re
from typing import List, Dict, Any, Optional
import spacy
from spacy.tokens import Doc, Span, Token

# Singleton holder for loaded spaCy NLP model
_NLP = None


def get_nlp():
    """Lazily loads the spaCy model once."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = spacy.blank("en")
            _NLP.add_pipe("sentencizer")
    return _NLP


class AtomicProposition:
    """
    Represents a decomposed proposition with relative clause offsets within the sentence.
    """
    def __init__(
        self, 
        text: str, 
        clause_type: str = "SIMPLE", 
        metadata: Optional[Dict[str, Any]] = None,
        clause_start: Optional[int] = None,
        clause_end: Optional[int] = None,
        offset_type: str = "EXACT"
    ):
        self.text = text.strip()
        self.clause_type = clause_type
        self.metadata = metadata or {}
        self.clause_start = clause_start
        self.clause_end = clause_end
        self.offset_type = offset_type

    def __repr__(self):
        return (
            f"AtomicProposition({self.text!r}, type={self.clause_type}, "
            f"span={self.clause_start}:{self.clause_end}, offset_type={self.offset_type})"
        )


def extract_named_entities(doc_or_span) -> List[Dict[str, str]]:
    """Extracts named entities from a spaCy Doc or Span."""
    entities = []
    if hasattr(doc_or_span, "ents"):
        for ent in doc_or_span.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_
            })
    return entities


# Regex for introductory rhetorical questions that convey embedded factual assertions
RHETORICAL_QUESTION_REGEX = re.compile(
    r"^(?:did you know|did you realize|have you heard|are you aware|can you believe|do you know)(?:\s+that)?\s+(.+?)\??$",
    re.IGNORECASE
)

# Regex to detect dummy/pleonastic 'it' which should not be replaced by coreference
PLEONASTIC_IT_REGEX = re.compile(
    r"^it\s+(?:is|was|seems|appeared|appears)\s+(?:important|clear|obvious|necessary|crucial|essential|worth|likely|unlikely|possible|impossible|said|believed|noted|recommended|expected|true|false)\b",
    re.IGNORECASE
)


def decompose_sentence(sentence_text: str, previous_subject: Optional[str] = None) -> List[AtomicProposition]:
    """
    Decomposes a single sentence into atomic propositions.
    
    Features:
    1. Unwraps rhetorical questions with embedded factual claims (e.g. "Did you know that X?").
    2. Semicolon decomposition into independent assertions.
    3. Decomposes compound predicates with coordinating conjunctions (and, but) with subject carry-over.
    4. Resolves third-person pronouns when referent is unambiguous, while avoiding pleonastic 'it'.
    5. Calculates relative clause spans (clause_start, clause_end) within the source sentence.
    """
    clean_sent = sentence_text.strip()
    if not clean_sent:
        return []

    # 1. Check for Rhetorical Questions with Embedded Factual Claims
    rhetorical_match = RHETORICAL_QUESTION_REGEX.match(clean_sent)
    if rhetorical_match:
        embedded_clause = rhetorical_match.group(1).strip()
        # Capitalize and punctuate to evaluate grammatical structure
        prop_text = embedded_clause[0].upper() + embedded_clause[1:]
        if not prop_text.endswith((".", "!", "?")):
            prop_text += "."

        # Verify that the embedded clause contains a verbal predicate (verb/aux)
        # e.g., "The Louvre houses..." has 'houses' (VERB), whereas "About the Moon." has no verb
        nlp = get_nlp()
        clause_doc = nlp(prop_text)
        has_predicate = any(t.pos_ in ("VERB", "AUX") for t in clause_doc)

        if has_predicate:
            clause_start_in_sent = rhetorical_match.start(1)
            clause_end_in_sent = rhetorical_match.end(1)

            return [AtomicProposition(
                text=prop_text,
                clause_type="EMBEDDED_FACTUAL_QUESTION",
                clause_start=clause_start_in_sent,
                clause_end=clause_end_in_sent,
                offset_type="EMBEDDED_CLAUSE",
                metadata={
                    "rhetorical_question": clean_sent,
                    "embedded_clause_slice": embedded_clause,
                    "extracted_from_rhetorical_question": True
                }
            )]

    # 2. Check for semicolon separation first (e.g. "Fact A; Fact B.")
    if ";" in clean_sent:
        parts = [p.strip() for p in clean_sent.split(";") if p.strip()]
        if len(parts) > 1:
            propositions = []
            curr_pos = 0
            for part in parts:
                part_clean = part
                if not part_clean.endswith((".", "!", "?")):
                    part_clean += "."

                part_idx = clean_sent.find(part, curr_pos)
                p_start = part_idx if part_idx != -1 else 0
                p_end = p_start + len(part)
                curr_pos = p_end

                sub_props = decompose_sentence(part_clean, previous_subject=previous_subject)
                for sp in sub_props:
                    sp.clause_start = p_start
                    sp.clause_end = p_end
                    sp.offset_type = "CLAUSE_SPAN"
                propositions.extend(sub_props)
            return propositions

    nlp = get_nlp()
    doc = nlp(clean_sent)

    # 3. Check for compound predicates joined by coordinating conjunctions (and, but, while)
    root = [token for token in doc if token.dep_ == "ROOT"]
    if root:
        root_token = root[0]
        conj_verbs = [t for t in root_token.conjuncts if t.pos_ in ("VERB", "AUX")]

        if conj_verbs:
            conj_verb = conj_verbs[0]
            # Find the coordinating conjunction connecting them (must appear after the root verb)
            cc_tokens = [
                c for c in doc 
                if c.dep_ == "cc" and c.text.lower() in ("and", "but", "while", "whereas")
                and c.i > root_token.i
                and (c.head == root_token or c.head == conj_verb or c.i < conj_verb.i)
            ]

            if cc_tokens:
                cc_idx = cc_tokens[0].i
                subjects = [t for t in root_token.children if "subj" in t.dep_]
                if subjects:
                    subj = subjects[0]
                    subj_subtree_span = doc[subj.left_edge.i : subj.right_edge.i + 1].text.strip()

                    clause1_tokens = doc[:cc_idx]
                    clause2_tokens = doc[cc_idx + 1 :]

                    clause1_text = clause1_tokens.text.strip()
                    clause2_text = clause2_tokens.text.strip()

                    # Clause 1 offsets (trim trailing whitespace before the conjunction)
                    c1_start = doc[0].idx
                    prev_tok = doc[cc_idx - 1]
                    c1_end = prev_tok.idx + len(prev_tok.text)

                    # Clause 2 offsets
                    c2_start = doc[cc_idx + 1].idx if len(doc) > cc_idx + 1 else c1_end
                    c2_end = doc[-1].idx + len(doc[-1].text)

                    # Check if clause 2 already has its own subject
                    has_own_subj = any("subj" in t.dep_ for t in clause2_tokens)
                    c2_offset_type = "EXACT"
                    if not has_own_subj and clause2_text:
                        # Prepend the subject to make it atomic
                        clause2_text = f"{subj_subtree_span} {clause2_text}"
                        c2_offset_type = "CLAUSE_SPAN"

                    if not clause1_text.endswith((".", "!", "?")):
                        clause1_text += "."
                    if not clause2_text.endswith((".", "!", "?")):
                        clause2_text += "."

                    if len(clause1_text.split()) >= 3 and len(clause2_text.split()) >= 3:
                        return [
                            AtomicProposition(
                                text=clause1_text, 
                                clause_type="COMPOUND_PREDICATE_1",
                                clause_start=c1_start,
                                clause_end=c1_end,
                                offset_type="EXACT",
                                metadata={"carried_subject": subj_subtree_span}
                            ),
                            AtomicProposition(
                                text=clause2_text, 
                                clause_type="COMPOUND_PREDICATE_2",
                                clause_start=c2_start,
                                clause_end=c2_end,
                                offset_type=c2_offset_type,
                                metadata={
                                    "carried_subject": subj_subtree_span,
                                    "original_clause_span": (c2_start, c2_end)
                                }
                            )
                        ]

    # 4. Simple Pronoun Resolution (Coreference) across sentences
    resolved_text = clean_sent
    resolved_metadata = {}
    offset_type = "EXACT"

    # Avoid resolving pleonastic 'it' (e.g. "It is important to note...")
    is_pleonastic = bool(PLEONASTIC_IT_REGEX.search(clean_sent))

    if previous_subject and not is_pleonastic:
        pronoun_match = re.match(r"^(It|They|He|She)\b\s+(.*)", clean_sent, re.IGNORECASE)
        if pronoun_match:
            pronoun = pronoun_match.group(1)
            remainder = pronoun_match.group(2)
            resolved_text = f"{previous_subject} {remainder}"
            offset_type = "PRONOUN_RESOLVED"
            resolved_metadata["resolved_pronoun"] = pronoun
            resolved_metadata["resolved_from_subject"] = previous_subject
            resolved_metadata["original_sentence"] = clean_sent

    return [AtomicProposition(
        text=resolved_text,
        clause_type="SIMPLE",
        clause_start=0,
        clause_end=len(clean_sent),
        offset_type=offset_type,
        metadata=resolved_metadata
    )]


def extract_salient_subject(text: str) -> Optional[str]:
    """Extracts the primary nominal subject from a proposition for cross-sentence coreference."""
    # Check pleonastic 'it' first
    if PLEONASTIC_IT_REGEX.search(text.strip()):
        return None

    nlp = get_nlp()
    doc = nlp(text)
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject_span = doc[token.left_edge.i : token.right_edge.i + 1].text.strip()
            if token.lower_ not in ("it", "they", "he", "she", "this", "that", "these", "those"):
                return subject_span
    return None
