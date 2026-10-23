# collection-server

A Cloudflare Worker at `collect.underfloor.ca` serves songs to the phone's
`#collect` page and stores the sessions it records. Both live in an R2 bucket
named `emg-collection`, and every request needs the bearer token. Set it up
once from this directory:

```sh
pnpm dlx wrangler r2 bucket create emg-collection
pnpm dlx wrangler secret put COLLECTION_TOKEN
pnpm dlx wrangler deploy
python3 publish_tracks.py
```

iOS does not play Vorbis, so `publish_tracks.py` converts each song in
`dashboard/tracks` to AAC. It keeps the `hard` level's note times and uploads
an index beside them.

`COLLECTION_TOKEN=... python3 pull_sessions.py` copies new sessions into
`dashboard/sessions`. They use the dashboard's layout, so `emg-tds/scripts`
reads them unchanged. The phone writes a window the band dropped as zeros with
its missing bits set, and an `emg_filler` event records the gap.

To work locally, put `COLLECTION_TOKEN=local-test-token` in `.dev.vars`. Then
`python3 publish_tracks.py --local` fills a local bucket for
`pnpm dlx wrangler dev --local`.
