# Northwind Engineering Handbook

## Code review

Every change ships through a pull request. A PR needs **one approving review**
from a code owner before it can merge. PRs that touch the `payments/` or
`auth/` directories require **two** approvals, at least one from the Security
guild.

We squash-merge to `main`. Direct pushes to `main` are blocked by branch
protection; the only exception is the release bot.

## Branching

- `main` is always deployable.
- Feature branches are named `feat/<ticket>-<slug>`, e.g. `feat/NW-1421-rate-limit`.
- Hotfix branches are `hotfix/<slug>` and may target the active release tag.

## Testing standards

New code must keep line coverage at or above **80%**. The CI gate fails the
build below that threshold. Unit tests run on every push; the full integration
suite runs nightly and on release candidates.

## On-call

Engineers join the on-call rotation after their first 90 days. A rotation is
one week, Monday 10:00 to the following Monday 10:00. The primary on-call must
acknowledge a page within **15 minutes**; after 15 minutes it escalates to the
secondary, and after 30 minutes to the engineering manager.

## Tech stack

Backend services are Python 3.12 (FastAPI). The primary datastore is
PostgreSQL 16; Redis is used for caching and rate limiting. Services are
containerized and deployed to Kubernetes.
