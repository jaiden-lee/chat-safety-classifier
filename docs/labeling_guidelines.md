# Labeling Guidelines: Personal-Information and Off-Platform Contact
Version: 1.1
Last Updated: 11/05/2026
Status: Draft (pending review)

These guidelines define the label for every example in the training, validation,
and test sets. They are written to be used in two places: by the LLM prompts
that generate (and label) synthetic conversations, and by humans reviewing
samples. When a case is unclear, add it to the
[decisions log](#7-borderline-decisions-log) instead of deciding it silently.

The version number above is stamped into every generated example
(`modeling.GUIDELINE_VERSION`). Any rule change bumps the version.

## 1. The labeling unit

Each example is **one target message**, judged together with the conversation
before it.

- **Causal view.** Use only the messages **before and including** the target.
  Never use later messages. The model sees the same view at serving time, so a
  label that depends on the future can't be learned.
- **Speaker tags.** Messages are tagged relative to the target's author:
  `[SELF]` is the author of the target, and `[U1]`, `[U2]`, … are the other
  participants in order of first appearance. Channel type is `[DM]` or `[GROUP]`.
- **The label is about the target message**, not the conversation. If contact
  info was shared three messages ago and the target is "lol ok", the target is
  negative.
- **One binary label.** `1` (positive) or `0` (negative). Tags (Section 5) are
  recorded alongside for evaluation slices but are not trained on.

## 2. The one-sentence rule

> A target message is **positive** if, given the conversation so far, it
> **shares** or **solicits** information that would let someone **contact or
> locate a person outside the platform**: an off-platform handle or link, a
> phone number, an email address, a specific physical location, or a request
> to move the conversation off the platform.

"The platform" means the game or app where the chat happens. Everything that
stays inside it (in-game names, friend requests, parties, guilds, in-game voice
chat) is on-platform and negative.

## 3. Positive: what counts

| ID | Rule | Examples (target message) |
|---|---|---|
| P1 | **Off-platform handle or link.** A username, tag, profile, server, or invite link for another service: Discord, Snapchat, Instagram, TikTok, Telegram, WhatsApp, Kik, X/Twitter, Twitch, YouTube channel, Reddit, Facebook, Signal, Steam/Xbox/PSN/Epic/Battle.net, and similar. | "my snap is jaydenxo", "discord.gg/k7Hq2", "add me on xbox: FrostByte221" |
| P2 | **Phone number.** Full or partial, in any format, once it's clear the digits are a phone number. Includes "text me at …" and WhatsApp numbers. | "555 201 8890", "my number starts with 415", "call me 4 1 5 …" |
| P3 | **Email address.** Personal email, in any format. | "its kayla.m09@gmail", "kayla dot m09 at gmail" |
| P4 | **Specific physical location.** Information that narrows a person down to a findable place: home address, street name, school name (any level), workplace, ZIP code, a specific landmark near home, or an invitation to meet in person. | "I go to Lincoln Middle", "I live on Maple Ave by the Target", "my zip is 90210" |
| P5 | **Solicitation.** Asking a chat participant for any P1–P4 information, or asking whether they use an off-platform service *in order to connect*. | "whats ur snap", "do you have discord?", "what school do u go to", "send me your number" |
| P6 | **Move request.** Asking or offering to continue the conversation off-platform, with or without naming where. Includes meeting in person. | "lets talk somewhere else", "text me", "dm me on insta", "hop in my discord call", "wanna meet at the mall saturday" |
| P7 | **Answer completed by context.** A message that has no telltale content by itself but completes a P1–P6 exchange: a bare username after "what's your discord", digits after "my number is", a reply of "same as here" to "what's your insta". The answer can come from any speaker. | `[U1] whats ur discord` → `[SELF] shadowfox_22` |
| P8 | **Third-party contact info.** Sharing or soliciting someone else's off-platform contact info. | "my cousin's snap is …", "whats your sister's insta" |
| P9 | **Off-platform self-promotion.** Pointing people to one's own off-platform profile, channel, or server. | "follow my twitch ttv/frostbyte", "join my server link in bio" |

A message that mixes benign content with any positive content is positive:
"gg that was close, add my snap jaydenxo".

## 4. Negative: what does *not* count

| ID | Rule | Examples (target message) |
|---|---|---|
| N1 | **On-platform social.** In-game usernames, friend requests, party, guild, or clan invites, in-game lobby or room codes, in-game voice chat. | "add me, my ign is FrostByte", "join my party", "lobby code 4KQ9", "get in vc" |
| N2 | **Talking about apps.** Mentioning another service as a topic, not as a way to connect. | "discord is down again", "tiktok is so cringe", "saw this on youtube" |
| N3 | **Public or official accounts.** The game's or a company's official accounts, support emails, public figures. | "report it to support@game.com", "the official discord has patch notes" |
| N4 | **Numbers that aren't contact info.** Scores, levels, coordinates, item or match IDs, prices, years, times, counts, and emergency numbers. | "I'm 2450 elo", "go to 120 -45", "match 7750218845", "call 911 if it's serious" |
| N5 | **Coarse location.** Country, state, province, city, region, or timezone, and questions about them. | "I'm from texas", "what timezone are u", "anyone else in canada" |
| N6 | **Refusals, warnings, and reports.** Declining to share, warning others, or describing someone else's request without passing on the info. | "I don't give out my snap", "don't share ur number here", "some guy asked for my discord lol" |
| N7 | **Content links.** Links to a video, clip, image, article, or wiki page that aren't a person's profile or a server invite. | "check this clip youtube.com/watch?v=…", "build guide on the wiki" |
| N8 | **Acknowledgments after an exchange.** Replies that neither share nor solicit new information, even when the context already contains a positive. | `[U1] my snap is jaydenxo` → `[SELF] ok` |
| N9 | **Signal-free fragments.** The first piece of split information when nothing yet shows it's contact info. The positive label goes on the message where the meaning becomes clear (Section 6.4). | `[SELF] 415` (first message of a split, with no prior hint) |
| N10 | **Other players' in-game names.** Referring to another player by their in-game username. | "xXslayerXx is cheating", "follow FrostByte he's carrying" |

**Out of scope (labeled negative):** age, gender, a real first name, a real full
name, appearance, and other personal details that aren't contact or location
information. These may become their own category later; for v1 they are
negative so the label stays focused on contact and location.

## 5. Tags (recorded, not trained on)

Every example carries tags so evaluation can be sliced (design doc §6.1, §6.6).
Tags are lists: use every value that applies.

### 5.1 Category (positives only)
| Tag | Covers |
|---|---|
| `handle` | P1, P7, P8, and P9 when the info is a handle or link |
| `phone` | P2, or P5/P7 aimed at a phone number |
| `email` | P3, or P5/P7 aimed at an email address |
| `location` | P4, or P5 aimed at a location |
| `platform_move` | P6, and P5 requests that don't name a specific identifier ("do you have discord?") |

### 5.2 Hard-positive tags (positives only)
| Tag | Definition | Example |
|---|---|---|
| `plain` | None of the tags below apply. Standard format, telltale keyword present. | "my snap is jaydenxo" |
| `obfuscated` | The information is deliberately altered to evade filters: spacing, character substitution or leetspeak, spelled-out digits, look-alike characters, "at/dot", inserted separators, reversal, or emoji standing in for letters or digits. | "five five five, two oh one…", "d1sc0rd", "j a y d e n x o" |
| `cross_message` | The label depends on another message in the context: a request answered later (P7), or information split across messages. A message that is positive by itself (a complete phone number, email, link, or a handle with its app named) is not `cross_message`, even when it answers a request. | `[U1] whats ur snap` → `[SELF] jaydenxo` |
| `implicit` | The target contains neither a platform name nor a recognizable identifier format; the meaning comes from euphemism or indirection. | "the yellow ghost app, same name as here", "gimme ur digits" |

### 5.3 Hard-negative types (negatives only)
| Tag | Covers |
|---|---|
| `easy` | Ordinary chat with nothing that resembles contact info |
| `number_lookalike` | N4: long or phone-shaped numbers that aren't contact info |
| `ingame_social` | N1: friend adds, party and guild invites, in-game names, lobby codes |
| `app_topic` | N2, N3, N7: other apps mentioned as a topic, official accounts, content links |
| `ingame_name_reference` | N10 |
| `coarse_location` | N5 |
| `refusal_warning` | N6 |
| `post_exchange` | N8 and N9: negatives right next to a positive exchange |

## 6. Decision procedure and tie-breakers

Apply in order; stop at the first rule that decides.

1. **Does the target contain, or complete, an off-platform handle or link, a
   phone number, an email, or a specific location** (P1–P4, P7–P9)? → positive.
2. **Does the target ask for any of those, or ask or offer to move off-platform**
   (P5, P6)? → positive.
3. Otherwise → negative. Pick the hard-negative type from Section 5.3.

### 6.1 "Would it work?" beats "is it real?"
Label by whether the information *could* be used to make contact, not whether
it's real. A made-up but plausible phone number is positive. We can't verify
identifiers, and neither can the model.

### 6.2 On-platform vs. off-platform
- In-game names, friend requests, and parties for the game the chat is in are
  **on-platform** (N1).
- Accounts on any other service are **off-platform** (P1), including console and
  PC networks (Steam, Xbox, PSN, Epic, Battle.net), because they come with their
  own messaging that the host platform doesn't moderate.
- "Add me" with no platform named is **on-platform by default**. It becomes
  positive only when context points off-platform (e.g., the previous message was
  "do you have discord").

### 6.3 Asking about a service
- Directed at a participant to connect ("do you have snap?", "u on discord?")
  → positive (P5, `platform_move`).
- A general topic ("does anyone still use kik", "is discord down for anyone")
  → negative (N2).

### 6.4 Split information
Information spread across several messages is labeled positive **from the first
message where it is clear the content is contact information**, and every later
message that adds a piece is also positive.
- `[SELF] my number is` → positive (announces sharing; P2).
- `[SELF] 415` (with the line above in context) → positive (`cross_message`).
- `[SELF] 415` as the very first message, with no prior hint → negative (N9).
  If the next message from the same speaker is "201 8890 text me", that
  message is positive (`cross_message`).

### 6.5 Location granularity
- Country, state, city, region, timezone → negative (N5).
- Street, address, school, workplace, ZIP code, "near the X on Y street",
  or an invitation to meet in person → positive (P4/P6).
- "Where are you from?" → negative. "What school do you go to?" or
  "what street do you live on?" → positive (P5).

### 6.6 Links
- Server invites, profile links, and channel links → positive (P1/P9).
- Content links (a specific video, clip, image, article, wiki page) → negative (N7).

### 6.7 Who the message is addressed to doesn't matter
In group chats, a positive is positive whether it's aimed at one person or at
everyone ("anyone wanna join my discord?").

## 7. Worked examples

Format: channel type, the context (oldest first), then the target marked `◀`.
Label, category, and tags follow.

### 7.1 Plain positives
**E1** `[DM]` → **1** · handle · plain
```
[U1]   gg that was fun
[SELF] ya add my snap jaydenxo09   ◀
```
P1: an off-platform handle with the platform named.

**E2** `[GROUP]` → **1** · phone · plain
```
[SELF] text me 415-201-8890   ◀
```
P2 and P6: a phone number and a move request.

**E3** `[DM]` → **1** · email · plain
```
[U1]   how do i send u the pics
[SELF] email them kayla.m09@gmail.com   ◀
```
P3.

**E4** `[GROUP]` → **1** · handle · plain
```
[SELF] join my server discord.gg/k7Hq2Lp   ◀
```
P1/P9: a server invite link.

**E5** `[DM]` → **1** · location · plain
```
[U1]   what grade are u in
[SELF] 7th at lincoln middle   ◀
```
P4: school name. (The grade alone would be out of scope.)

**E6** `[DM]` → **1** · platform_move · plain
```
[U1]   this game chat is so laggy
[SELF] lets just talk on discord   ◀
```
P6: a move request that names the platform.

**E7** `[DM]` → **1** · handle · plain
```
[SELF] add me on xbox FrostByte221   ◀
```
P1: console networks are off-platform (Section 6.2).

### 7.2 Solicitation
**E8** `[DM]` → **1** · handle · plain
```
[SELF] ur really good
[U1]   thx
[SELF] whats ur snap   ◀
```
P5. The asker wrote the target, so the asker is `[SELF]`.

**E9** `[DM]` → **1** · platform_move · plain
```
[SELF] u have discord?   ◀
```
P5: asking a participant whether they use a service in order to connect (Section 6.3).

**E10** `[DM]` → **1** · location · plain
```
[U1]   im in 6th grade
[SELF] same what school   ◀
```
P5: asking for a school.

**E11** `[DM]` → **1** · platform_move · implicit
```
[SELF] we should talk somewhere more private   ◀
```
P6: a move request with no platform named and no keyword.

**E12** `[DM]` → **1** · location · plain
```
[U1]   we live pretty close lol
[SELF] wanna meet up at the mall saturday?   ◀
```
P6: an invitation to meet in person.

### 7.3 Cross-message positives
**E13** `[DM]` → **1** · handle · cross_message
```
[U1]   whats ur discord
[SELF] shadowfox_22   ◀
```
P7: a bare username that answers a solicitation.

**E14** `[GROUP]` → **1** · handle · cross_message
```
[U1]   anyone got insta
[U2]   me
[U1]   whats ur @
[SELF] hers is lily.k.03   ◀
```
P7 and P8: a third speaker answering with someone else's handle.

**E15** `[DM]` → **1** · phone · cross_message
```
[SELF] my number is
[SELF] 415   ◀
```
Section 6.4: the earlier message makes it clear these digits are a phone number.

**E16** `[DM]` → **1** · phone · cross_message
```
[SELF] 415
[U1]   ?
[SELF] 201 8890 text me   ◀
```
Section 6.4. The earlier "415" message by itself would be negative (E37).

**E17** `[DM]` → **1** · handle · cross_message, implicit
```
[U1]   do u have insta
[SELF] ya same name as here   ◀
```
P7: "same name as here" plus the in-game name the asker already knows identifies the account.

**E18** `[DM]` → **1** · handle · cross_message
```
[U1]   add me on snap
[SELF] whats ur name on there   ◀
```
P5: a solicitation in a context that's already off-platform.

### 7.4 Obfuscated positives
**E19** `[DM]` → **1** · phone · obfuscated
```
[SELF] four one five two oh one eight eight nine oh   ◀
```
Spelled-out digits.

**E20** `[DM]` → **1** · handle · obfuscated
```
[SELF] add me on d1sc0rd its sh4d0wf0x   ◀
```
Leetspeak.

**E21** `[DM]` → **1** · handle · obfuscated
```
[SELF] s n a p : j a y d e n x o   ◀
```
Spacing.

**E22** `[DM]` → **1** · email · obfuscated
```
[SELF] kayla m09 at gmail dot com   ◀
```
"at/dot".

**E23** `[DM]` → **1** · phone · obfuscated
```
[SELF] 4.1.5..2.0.1..8.8.9.0   ◀
```
Inserted separators.

**E24** `[DM]` → **1** · handle · obfuscated, implicit
```
[SELF] my 👻 is jaydenxo   ◀
```
Emoji standing in for the platform name. No keyword is present, so it's also `implicit`.

**E25** `[DM]` → **1** · handle · obfuscated
```
[SELF] tele: @ⅽhаrlіе_k   ◀
```
Look-alike (Cyrillic and Roman-numeral) characters inside the handle.

### 7.5 Implicit positives
**E26** `[DM]` → **1** · phone · implicit
```
[SELF] gimme ur digits   ◀
```
A slang solicitation for a phone number.

**E27** `[DM]` → **1** · handle · implicit
```
[SELF] the yellow ghost app, same name as here   ◀
```
Snapchat by description.

**E28** `[DM]` → **1** · handle · implicit
```
[SELF] my name's the same on every app   ◀
```
Tells the reader how to find them off-platform.

### 7.6 Hard negatives
**E29** `[GROUP]` → **0** · ingame_social
```
[SELF] add me my ign is FrostByte221   ◀
```
N1: an in-game name with no off-platform context. (Compare E7.)

**E30** `[GROUP]` → **0** · ingame_social
```
[U1]   need 1 more
[SELF] join lobby 4KQ9-77TX   ◀
```
N1: lobby code.

**E31** `[GROUP]` → **0** · number_lookalike
```
[SELF] bro my score was 4152018890 how   ◀
```
N4: a phone-shaped number in a score context.

**E32** `[GROUP]` → **0** · number_lookalike
```
[SELF] meet at 120 -45 near the castle   ◀
```
N4: in-game coordinates (in-game "meet" is on-platform).

**E33** `[GROUP]` → **0** · app_topic
```
[SELF] is discord down for anyone else   ◀
```
N2 (Section 6.3).

**E34** `[GROUP]` → **0** · app_topic
```
[SELF] patch notes are on the official discord   ◀
```
N3.

**E35** `[GROUP]` → **0** · coarse_location
```
[U1]   where u from
[SELF] texas, u?   ◀
```
N5.

**E36** `[DM]` → **0** · refusal_warning
```
[U1]   whats ur snap
[SELF] i dont give that out sorry   ◀
```
N6: a refusal. (The solicitation before it is positive.)

**E37** `[DM]` → **0** · post_exchange
```
[SELF] 415   ◀
```
N9: the first fragment, with nothing yet showing it's a phone number.

**E38** `[DM]` → **0** · post_exchange
```
[U1]   my snap is jaydenxo09
[SELF] ok   ◀
```
N8: an acknowledgment.

**E39** `[GROUP]` → **0** · ingame_name_reference
```
[SELF] xXslayerXx is cheating report him   ◀
```
N10.

**E40** `[GROUP]` → **0** · app_topic
```
[SELF] check this clip youtube.com/watch?v=dQ8x2   ◀
```
N7: a content link, not a profile.

**E41** `[GROUP]` → **0** · refusal_warning
```
[SELF] if anyone asks for ur number here just report them   ◀
```
N6: a warning.

**E42** `[GROUP]` → **0** · easy
```
[U1]   gg
[SELF] gg wp everyone   ◀
```

## 8. Borderline decisions log

Decisions made explicitly so the data stays consistent. Each entry gives the
ruling and the reason. Revisit during the Step 4 audit.

| # | Case | Ruling | Rationale |
|---|---|---|---|
| B1 | In-game usernames and in-game friend adds | Negative | On-platform; the host platform still moderates the conversation (design doc §10.2). |
| B2 | Steam / Xbox / PSN / Epic / Battle.net IDs | Positive | Each has its own messaging the host platform can't moderate, which is the risk we're targeting. |
| B3 | "Add me" with no platform named | Negative unless context points off-platform | Default to the host platform; otherwise the most common in-game phrase becomes a false block. |
| B4 | "Do you have X?" directed at a participant | Positive (`platform_move`) | It's the opening move of nearly every off-platform exchange; waiting for the handle makes cross-message recall harder. |
| B5 | ZIP code | Positive | Narrows to a few thousand households; for minors, "what's your zip" is a meaningful risk signal. |
| B6 | Invitations to meet in person | Positive (`platform_move`) | Leaving the platform physically is at least as serious as leaving it digitally. |
| B7 | Plausible-but-fake identifiers | Positive | Label what it could be used for (Section 6.1); neither labelers nor the model can verify. |
| B8 | Acknowledgments after an exchange ("ok", "added u") | Negative | The label is for the target, not the conversation. "added u" and "sent u a request on snap" share and ask for nothing new. |
| B9 | Real full name | Negative (out of scope v1) | Would be positive in principle, since a full name can be used to find someone off-platform. It's excluded from v1 to keep the scope to G1's contact and location categories; a candidate for the first scope extension. |
| B10 | First fragment of a split with no signal | Negative | Causal rule: the label can't depend on future messages. |
| B11 | Content links vs. profile/server links | Content negative, profile/server positive | Only the second lets someone reach the person. |

## 9. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 11/04/2026 | Initial draft. |
| 1.1 | 11/05/2026 | `cross_message` excludes messages that are positive by themselves (§5.2). |
