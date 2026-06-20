# Northwind Deployment Runbook

## Release cadence

We ship to production **twice a week**, on Tuesday and Thursday at 14:00 UTC.
Emergency hotfixes can go out any time with VP Engineering sign-off.

## Promotion flow

A build is promoted through three environments in order: **dev → staging →
production**. A build must soak in staging for at least **2 hours** with no
SEV-1 or SEV-2 alerts before it is eligible for production.

## Deploy steps

1. Cut a release branch and tag it `vX.Y.Z`.
2. CI builds the image and pushes it to the registry.
3. Argo CD syncs the staging cluster automatically.
4. Run the smoke suite: `make smoke ENV=staging`.
5. After the 2-hour soak, approve the production sync in Argo CD.
6. Production rollout is **canary**: 10% of traffic for 15 minutes, then 100%.

## Rollback

If error rate exceeds **2%** or p95 latency exceeds **800 ms** after a deploy,
roll back immediately by re-syncing Argo CD to the previous tag. Rollback
should complete within **5 minutes**. File a SEV-2 incident and start a
post-mortem within 24 hours.

## Database migrations

Migrations are applied **before** the app rollout using a pre-sync hook. All
migrations must be backward compatible (expand-then-contract); never drop a
column in the same release that stops writing to it.
