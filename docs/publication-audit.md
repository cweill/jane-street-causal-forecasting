# Publication audit

The repository remains private; changing GitHub visibility is the owner's action.
The working tree now contains only the Patrick Yam reconstruction and its research
harness. Removed implementations remain in earlier commits so recorded experiments
can still be reproduced. Historical source fingerprints are not rewritten.

## Secrets and private data

Gitleaks 8.30.1 scanned all 45 reachable commits through `2053ee7`, before the cleanup,
with redacted output. Both findings were reviewed against their generating code:

- `patrick-submission-rehearsal.json`: a random UUID echoed by a job-lifetime probe,
  generated with `uuid.uuid4().hex`; it is not an authentication token.
- `patrick-ol-interruption.json`: a content-addressed preprocessing-cache hash,
  not an API key.

No confirmed credentials were found. `.gitleaksignore` suppresses only these two
exact historical fingerprints, not files or whole classes of keys. GitHub Actions
now scans fetched history with a pinned, checksum-verified Gitleaks binary.

No raw competition parquet, model weights, private keys, `.env` files, Kaggle
credentials, or `.netrc` files are tracked or present in the scanned file history.
The ignore rules cover credentials, data exports, weights, and local run outputs.
The Modal upload code uses explicit project-directory allowlists. W&B credentials
are read from local configuration or an authenticated Modal Secret, never hardcoded.
Tracked text contains no personal absolute `/Users/...` paths or embedded URL
credentials. Slide-image metadata contains dimensions, resolution, and screenshot
annotations; plot metadata identifies Matplotlib. No location or account metadata
was found there.

## Metadata that publication will expose

Git commits retain the author's name and email. Documentation retains the GitHub,
W&B, and Modal account/project links, run/call identifiers, dataset hashes, aggregate
scores and timing/resource measurements. These support provenance and are not
credentials, but they identify the account and research activity. W&B and Modal
access controls are separate from GitHub repository visibility.

Changing visibility also exposes earlier commits, including removed code and old
experiment records. This cleanup does not rewrite history or anonymize authorship.
The scan covers Git contents and reachable history, not a guarantee about external
service permissions or material added after this audit.

## Attribution and reproducibility

The code is an independent reconstruction with documented assumptions. Public-talk
slide excerpts retain their source attribution; the repository does not include the
competition dataset or the author's submission code. Historical runs use their
recorded source commits; fresh runs require their own data and authenticated compute
accounts. See the README for tested commands and completed result summaries.
