# Future Work: Speech Keyword Detection (deferred from Week 7.6b)

**Status:** Deferred. No code shipped. Design preserved for future implementation.

**Why deferred:** No labeled audio dataset of emergency phrases is
available; TTS-generated audio would violate the project's
honest-evaluation principle; self-recording 48–100 clips was not
feasible in the current session.

**What the capability would be:** Given a short audio clip, detect
whether it contains a spoken request for emergency assistance
("call the police", "help me", "there's a fire"). Two-baseline
approach:

1. Rule-based: Whisper-tiny transcription → phrase matcher
2. Zero-shot NLI: Whisper transcription → BART-MNLI classification
   into {emergency, non-emergency, ambiguous}

**What it would need:**
- 100+ labeled audio clips (50 emergency + 50 control, half spoken, half shouted)
- Or 200+ labeled *transcripts* (generated via LLM, no audio) for the NLI approach

**What it would prove:** Whether pretrained speech-to-text + a
classifier can detect emergency intent better than keyword rules alone.

**What it would NOT prove:** Real-world emergency detection capability.
Shouted speech in noise is out of scope; documented as a limitation.

**Resume line (deferred):** *"Designed a multi-baseline emergency-phrase
detector (Whisper + rule-based / zero-shot NLI), deferred pending a
labeled audio corpus."*
