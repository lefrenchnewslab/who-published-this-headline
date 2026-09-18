# Tier-2 Gold Codebook — framing-measure validation (v2, operationally exact)

Annotate `tier2_gold_packet.csv`. Fill the five `gold_*` columns; **do not rename columns or reorder rows**.
Save as `tier2_gold_annotated.csv` in the same folder, then run the validation cell.

**This codebook is the single source of truth for what each measure means.** The notebook extractor
(`tier2_matched_event_framing_v2.ipynb`, cell 10 `feats_from_doc`) is written to implement *exactly* these
definitions. When the gate disagrees, we fix whichever side is wrong against this document — codebook and
extractor must always encode the same rule. **Version this file; never change a definition without also
changing the extractor in the same commit.**

### How to code
1. Judge the **headline text alone**. `category` tells you the desk; you do not need the article.
2. Code the **construct as defined here**, not your guess of what the parser did, and not your opinion of the topic.
3. Apply each axis's **numbered decision procedure in order** and stop at the first rule that fires.
4. Leave a cell blank **only** if the headline is truncated to genuine nonsense (see §0). Otherwise always code.
5. When two readings remain after the procedure, apply the axis's **tie-breaker**, then move on.

### Conventions used below
- **CLAUSE** = a finite or infinitival complement that states *what was said/claimed* (a proposition): the
  grammatical pattern *« X dit / estime / affirme **que** … »*, *« X annonce **vouloir** … »*, or a **direct quote**.
- **ACTION/EVENT object** = a noun phrase naming a thing done or happening, not a reported proposition
  (*« annonce **un congrès** »*, *« réclame **l'extradition** »*, *« adopte **la loi** »*).
- **SOURCE** = an identifiable person, institution, or named/standing collective to whom words are attributed.

---

## §0 · Truncated headlines

~20% of pool headlines are cut mid-sentence at the scrape (`…en décemb`, `…avant la signa`). These are **not**
nonsense — code them on what is **visible and complete**:
- If the part you can see fully determines the axis, code it normally.
- If the cut removes the only evidence for a positive label (e.g. the reporting clause is cut off), code **0**
  for that axis (absence), **not** blank.
- Blank only if the visible fragment is uninterpretable.

*(We will separately re-validate on truncation-free headlines; consistent coding of the visible text is what
matters here.)*

---

## §1 · `gold_valence` — affective slant toward the event   (−1 / 0 / +1)

**Construct:** the slant a typical reader perceives in *how the event is framed*. Not your opinion of the topic.

**Decision procedure**
1. Is the **event itself** clearly adverse (death, crisis, scandal, conflict, sanction, job losses, fine,
   condemnation, accident)? → **−1**, even if reported in flat factual terms.
2. Is it clearly favourable (success, win, record, agreement, relief, tribute, rescue, acquittal)? → **+1**.
3. Mixed signals → code the **dominant** slant; if genuinely balanced → **0**.
4. Plain factual report with no evaluative loading and a neutral event → **0**.

**Tie-breaker:** if you cannot decide between 0 and ±1, ask "would most readers feel this is good news or bad
news for the people in it?" If still unclear → **0**.

| Headline (abridged) | label | why |
|---|---|---|
| Isabelle Balkany exclue définitivement de la Légion d'honneur | −1 | sanction |
| Deux pompiers tués dans une embuscade | −1 | deaths |
| Audi : 7500 suppressions d'emplois prévues d'ici 2029 | −1 | job losses |
| Uber condamné à verser plusieurs millions d'euros | −1 | penalty |
| L'interdiction de TikTok levée en Nouvelle-Calédonie | +1 | a ban lifted = relief |
| Benalla et son épouse libérés, sans poursuites | +1 | release/relief |
| Sophia Chikirou candidate de LFI à Paris | 0 | neutral announcement |
| Retraites : la mission de la Cour des comptes sera plus large que prévu | 0 | factual, no slant |

*(Extractor outputs a continuous score; validated by Spearman ρ on the ordering, not exact match.)*

---

## §2 · `gold_attribution` — is the headline's information SOURCED to a speaker?   (1 / 0)

**Construct:** the headline presents a **claim, statement, opinion, judgement, or stance as coming from an
identifiable source**, rather than asserting it in the outlet's own neutral voice. The test is *reported
content*, not merely "a person did a verbal thing."

**Decision procedure** (stop at first hit → 1; if none → 0)
1. **`selon / d'après / pour X`** frame present? → **1**.
2. A **direct quote that is reported speech** — the quoted span contains a **finite verb** (or first/second-person
   speech), tied to a speaker — *« Une belle page que nous avions commencé à écrire »*? → **1**.
   *(A quoted span with **no** verb is a scare/descriptive term — « espion », « une fausse solution »,
   « relation amoureuse non déclarée » — that hedges (§5) but does **not** by itself create attribution. So
   « Macron annonce l'objectif de « développer un réseau… » » is **0**.)*
3. A **SAY-class verb** (see lexicon) whose subject is the source **and which introduces a complement clause**
   — recognise it by the complementizer ***que / qu'*** (*« estime **que** … »*) or an inverted quotative
   (*« …, **annonce** l'Élysée »*, *« …, **juge** le LR Geoffroy Didier »*)? → **1**.
4. A **STANCE-class verb** (dénoncer, accuser, fustiger, saluer, regretter, déplorer, alerter, se dire +adj)
   whose subject is the source voicing the stance about something? → **1**.
   *(These report a communicative position even when they govern a noun: « les Palestiniens dénoncent les
   attaques » = the Palestinians' voiced condemnation.)*
5. Otherwise → **0**.

**Hard exclusions (these are 0, even though they contain a "verbal" verb):**
- **Announcing / committing one's own future ACTION or EVENT** with no reported proposition:
  *« Faure annonce un congrès »*, *« se déclare candidat »*, *« Hamon annonce quitter la politique »*,
  *« renonce à vendre M6 »*. (verb governs an action/event, not a claim)
- **Demands/requests naming an action** with no clause: *« Darmanin réclame l'extradition »*,
  *« demande la démission »*. → **0**. *(If it governs a clause — « demande **que** … » — code 1.)*
- **Non-communicative actions:** *« le gouvernement adopte la loi »*, *« Boeing va verser 2,5 Mds »*,
  *« Uber condamné »* (judicial action, not the outlet sourcing a claim).

> **Resolved boundary (changed from earlier coding):** "X réclame/annonce/demande + *a noun*" is **0**.
> Attribution requires reported propositional content (a clause/quote) **or** `selon`/`pour X` **or** a
> stance-verb voicing a position. This is the single rule that previously made the gold internally
> inconsistent ("annonce un congrès"→0 vs "réclame l'extradition"→1); we now code both **0**.

| Headline (abridged) | label | why |
|---|---|---|
| le parquet italien **estime que** Meta doit 887 M€ de TVA | 1 | SAY-verb + clause |
| Joe Biden "apte" à assurer ses fonctions, **selon son médecin** | 1 | selon X |
| nationalisation "une fausse solution", **juge** le LR Geoffroy Didier | 1 | SAY-verb + quote |
| Bayrou **dit n'avoir** « jamais été informé » | 1 | SAY-verb + clause/quote |
| les Palestiniens **dénoncent** les attaques de colons | 1 | STANCE-verb voicing position |
| Pour les JO, les transports ne seront "pas prêts", **prévient** Hidalgo | 1 | SAY-verb + quote |
| Faure **annonce un congrès** « en juin » et **se déclare** candidat | 0 | own action; « en juin » is a 1–2-tok scare quote, not reported speech |
| Darmanin **réclame** aux Émirats **l'extradition** d'une quinzaine de trafiquants | 0 | demand + noun, no clause |
| Le ballon chinois **abattu** par l'armée américaine | 0 | outlet states a fact, no source |
| Bertelsmann **renonce** à vendre M6 | 0 | own action |

**SAY-class verbs (clausal):** affirmer, déclarer, dire, estimer, juger, assurer, soutenir, expliquer,
indiquer, préciser, annoncer (que/+inf), prévenir, prévoir (que), confirmer, démentir, reconnaître,
répondre, rétorquer, répliquer, marteler, insinuer, suggérer, promettre (que).
**STANCE-class verbs (voiced position, noun object OK):** dénoncer, accuser, condamner (verbalement),
fustiger, saluer, regretter, déplorer, alerter, plaider, se dire/se déclarer + adjectif.
**Excluded from both (action/event verbs):** réclamer, demander, proposer, renoncer, adopter, voter,
verser, nommer, condamner (judiciairement), licencier, signer, lancer — unless they govern a clause.

---

## §3 · `gold_passive` — is the main predication passive?   (1 / 0)

**Construct:** the grammatical subject **undergoes** the main action. Includes **reduced/elliptical passives**
common in headlines (past participle predicated of the subject, no finite *être*), with or without a *par*-agent.

**Decision procedure**
1. Can you insert *« a été / ont été »* before the participle and it reads naturally
   (*« le ballon [a été] abattu »*, *« Castaner [a été] nommé »*)? → **1**.
2. *par / par le / par les* introducing the agent of a participle? → **1**.
3. Otherwise → **0**.

**Hard exclusions (these are 0 — they look participial but are NOT passive):**
- **être + intransitive perfect** (the participle is the verb's own past, not a passive):
  *« est mort »*, *« est décédé »*, *« est arrivé »*, *« est tombé »*, *« est devenu »*. → **0**.
- **Reflexive / pronominal:** *« un incendie s'est déclaré »*, *« le scénario se précise »*,
  *« se sépare de »*. → **0**. *(Exception: clearly passive-meaning pronominals like « se voit infliger une
  amende » → 1, subject undergoes.)*
- **Verbless equatives with no action:** *« Bayrou, un premier ministre fragilisé »* → 0.

| Headline (abridged) | label | why |
|---|---|---|
| Castaner **nommé** président du conseil d'administration | 1 | reduced passive (a été nommé) |
| Jovenel Moïse **assassiné** | 1 | reduced passive |
| L'interdiction de TikTok **levée** | 1 | reduced passive |
| le ballon espion chinois **abattu par** les États-Unis | 1 | passive + par-agent |
| une boîte noire du Boeing accidenté **envoyée** aux États-Unis | 1 | reduced passive |
| Shein **se voit infliger** une amende record | 1 | pronominal passive, subject undergoes |
| Claude Bébéar… **est décédé** | 0 | être + intransitive perfect |
| Roland Dumas **est mort** à 101 ans | 0 | être + intransitive perfect |
| un incendie **s'est déclaré** dans la Bourse | 0 | reflexive, not passive |
| Les États-Unis **abattent** le ballon | 0 | active |

---

## §4 · `gold_agent_subject` — is the subject a real-world ACTOR?   (1 / 0)

**Construct:** the grammatical subject (or, in a verbless/nominal headline, the head noun in subject position)
is an **agent** — an entity capable of intentional action — performing the action in **active voice**.
**0** if the subject is a patient/undergoer, an event, a statistic, or an inanimate/abstract thing.

**An ACTOR is:** a named person or organisation (PROPN / PER / ORG); **or** a common-noun human/institutional
collective — see the ACTOR-NOUN list; **or** a demonym/plural human group (*les Palestiniens, les démocrates,
les syndicats*).

**Decision procedure**
1. Is the main verb **passive** (§3 = 1)? → subject undergoes → **0**.
2. Is the subject's head an **unaccusative / happening** verb (mourir, décéder, exister, augmenter, chuter,
   se produire, dérailler, s'effondrer…)? The subject is an undergoer → **0**.
3. Is the subject an **event, abstraction, count, or inanimate** (*la mission, le scénario, l'inflation,
   l'alliance, la charte, un incendie, trois morts, le titre*)? → **0**.
4. Is the subject an **ACTOR** (above) and the verb an **active action** it performs? → **1**.
5. Verbless/nominal headline with no action predicated → **0** (no agent role exists).

**Tie-breaker:** if the subject is a person but is the **target** of the action (*licencié, exclu, tué,
condamné, nommé*), that is passive/undergoer → **0**.

| Headline (abridged) | label | why |
|---|---|---|
| **le gouvernement** relève le plan Vigipirate | 1 | institutional actor, active |
| **syndicats et patronat** se disent prêts à une revalorisation | 1 | collective actors, active |
| les **Palestiniens** dénoncent les attaques | 1 | human group, active |
| les **démocrates** à un siège de la majorité | 1 | human group (verb implied active) |
| **Sophia Chikirou** officialise sa candidature | 1 | named person, active |
| **la cathédrale de Reims** s'en remet au e-commerce | 1 | institution acting (metonymic actor) |
| **la mission** de la Cour des comptes sera plus large | 0 | abstraction, copular |
| **le scénario** d'une réforme se précise | 0 | abstraction + reflexive |
| **l'inflation** réaccélère à 5,9% | 0 | abstraction/statistic |
| le ballon chinois **abattu** par… | 0 | passive, patient subject |
| **François Hommeril** licencié | 0 | person is the target |
| **Trois morts** dans un incendie | 0 | count/event |

**ACTOR-NOUN list (common-noun actors that count as agents):** gouvernement, État, ministère, ministre,
président, maire, préfet, parquet, justice, tribunal *(as actor, e.g. « le tribunal condamne »)*, police,
gendarmerie, armée, syndicat(s), patronat, parti, opposition, majorité, entreprise, groupe, société, banque,
direction, conseil, commission, assemblée, sénat, Parlement, Union (européenne), Commission, club, équipe,
fédération, association, ONG, église *(institution)*, manifestants, salariés, grévistes, électeurs,
habitants, + demonyms/plural human groups (les Russes, les Palestiniens, les démocrates, les Français…).

---

## §5 · `gold_hedging` — does the headline DISTANCE the claim from established fact?   (1 / 0)

**Construct:** the headline signals the proposition is **not asserted as settled fact** — it is hypothetical,
attributed-not-endorsed, or flagged with distancing quotes.

**Decision procedure** (any hit → 1)
1. **Epistemic modality / conditional:** *pourrait, peut, peut-être, probablement, sans doute, semble,
   paraît, risque de*, or **conditional mood** (*serait, aurait, devrait* in « X serait impliqué »,
   « aurait menti »). → **1**.
2. **Evidential distancing:** *selon / d'après / pour X* (outlet attributes rather than endorses). → **1**.
   *(So `selon X` is BOTH attribution and hedging.)*
3. **Scare/distancing quotes:** quotation marks around a **single contested word or short phrase
   (≤ ~4 tokens)** that the outlet won't assert in its own voice — *le ballon « espion »*, Biden « apte »,
   Le Pen « fatigué », une « fausse solution ». → **1**.
4. Otherwise → **0**.

**Hard exclusions (0):**
- **Full-sentence direct quotes** (the whole headline, or a long clause, in quotes) — that is *attribution*,
  not hedging, unless the quoted span is a short contested term per rule 3.
  *« "Ils ont agi en toute impunité" : … »* → the quote is a full clause → hedging **0** (attribution 1).
- A categorical factual statement with no modal/evidential/scare-quote — *« Le ballon abattu »*,
  *« Hommeril licencié »* → **0**.

**Tie-breaker for quotes:** quote **one short term** = scare quote → 1. Quote **a whole sentence/long clause**
= reported speech → 0 (that lives in attribution).

| Headline (abridged) | label | why |
|---|---|---|
| Biden « **apte** » à assurer ses fonctions, selon son médecin | 1 | scare quote + selon |
| Jean-Marie Le Pen, 95 ans et « **fatigué** », placé sous protection | 1 | short scare quote |
| nationalisation « **une fausse solution** » | 1 | short contested phrase |
| le ballon « **espion** » chinois | 1 | one contested word |
| X « **serait** impliqué » dans l'affaire | 1 | conditional |
| la réforme **pourrait** être reportée | 1 | epistemic modal |
| « **Ils ont agi en toute impunité** » : les Palestiniens dénoncent… | 0 | full-clause quote = attribution, not hedge |
| Le ballon chinois abattu par les États-Unis | 0 | categorical, no hedge |

---

## §6 · Cross-axis interactions (code each independently)

- **`selon X`** → attribution **1** and hedging **1**. Agent_subject judged on the matrix-clause subject.
- A **passive** (§3=1) almost always forces **agent_subject 0** (subject is patient).
- A **full attributed sentence** ("…", affirme X) → attribution **1**, hedging **0** (unless a short scare
  quote is also present); agent_subject = is **X** (the speaker) the active subject? usually yes → 1.
- **Stance verbs** (dénoncer/accuser…) → attribution **1** and, with an active human subject, agent_subject **1**.

---

## §7 · Admission gate
A measure enters the confirmatory analysis only if it clears **κ ≥ 0.60** (binary: attribution, passive,
agent_subject, hedging) or **ρ ≥ 0.50** (valence). The paper's framing claim rests on **attribution** and
**agent_subject** — those two clearing the bar is the priority; valence is reported as instrument-limited
regardless. For inter-annotator agreement, a second rater independently codes a ~40-row overlap; report
Cohen's κ to bound the human ceiling against which the extractor κ is read.

---

## §8 · Changelog
- **v2.2 (2026-06-23):** Second gate run (κ had regressed to 0.687 — the v2.1 finite-verb quote rule fired on
  headline/illustrative quotes with **no speaker**). Fixes: (A) a quote counts as attribution only when a
  **reporting cue** is present (SAY/STANCE verb or `selon`) — bare/scare/illustrative quotes no longer trigger;
  (B) added **surface-regex fallbacks** (reporting-verb form + `que`; comma/quote + reporting verb + capitalized
  source) so detection survives spaCy POS/`nsubj` mis-tags (fixes "estime que Meta", "…annonce l'Élysée",
  "…juge le LR Geoffroy Didier"); (C) REPORT verbs (dire/affirmer/…) now count with an **infinitival** clause
  (fixes "Bayrou dit n'avoir…"); (D) "saluer la mémoire/mort" excluded as a tribute idiom (FP). **Gold
  override:** "Brandt : un projet… prévoit de sauver 150 emplois" gold_attribution 1→0 (subject *un projet* is
  not a speaker; no SAY+clause-with-source, no `selon`, no quote).
- **v2.1 (2026-06-23):** Attribution refinements from the first gate run (κ 0.561→0.749). Quote rule now keys
  on a **finite verb inside the quote** (reported speech) rather than token count — descriptive scare quotes
  ("une fausse solution", Macron's "développer…") no longer create attribution. Clause detection keys on the
  **`que/qu'` complementizer** (label-independent, fixes "estime que Meta…") and on **inverted quotatives**
  ("…, annonce l'Élysée"). Known residuals: a quote that is reported speech but has no finite verb (Bayrou
  "dit n'avoir jamais été informé") may read 0; "salue la mémoire" (tribute idiom) may read 1.
- **v2 (2026-06-23):** Operationalised every axis with numbered decision procedures, include/exclude lists,
  closed verb/actor lexicons, and few-shot tables built from the actual error cases. **Resolved boundaries
  that made v1 gold internally inconsistent:** (a) attribution now requires reported propositional content
  (clause/quote) or `selon`/stance-verb — "X réclame/annonce + noun" is **0** (was sometimes 1); (b) passive
  explicitly excludes être-intransitive perfect (*est mort/décédé*) and reflexives; (c) agent_subject
  explicitly **includes** common-noun/collective actors via the ACTOR-NOUN list and **excludes**
  abstractions/events; (d) hedging scare-quote rule restricted to short contested spans (≤~4 tokens), full
  quotes routed to attribution. Notebook `feats_from_doc` synced in the same pass.
