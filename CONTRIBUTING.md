# Contributing to lapstack

Bug reports, questions and patches are welcome as GitHub issues and pull
requests. There is no template to fill in and no agreement to sign: by
sending a change you agree that it is yours to give and that it is offered
under the MIT license like the rest of the repository.

## A report we can act on

What you did, what you expected, what happened instead — a sentence each is
plenty — plus the operating system, the browser or build you ran, and the GPU.
A few frames that reproduce the problem help most of all; a link to a shared
folder beats an attachment, since raw files are large. Nothing in lapstack
reports back to us, so if it broke, we only know when you tell us.

## Building

`just` lists the developer commands; the README's *Build & run* section has
the plain cargo invocations, and `web/build.sh` builds the browser app
(needs the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` 0.2.128).

```
cargo build --release                       # the CLI and the raw decoder module
cargo test --release -p lapstack-core       # the core's unit tests
just build-web && just serve                # the browser app on http://localhost:8765/
node web/test/headless.mjs --align          # the browser end-to-end test (needs a GPU and Chrome)
```

CI runs the native build and tests on Linux, Windows and macOS and the wasm
build on every push. The tree is not clippy- or rustfmt-clean; do not reformat
code you are not otherwise changing.

## Where things are

The README is the design document: every formula the engine implements is
stated there with the deviations from the papers called out, and each section
names the file it describes. `docs/README.md` cites the papers. The browser
app's vocabulary (LAP, DFR, Focus depth, Winner) and layout rules are settled;
a new control goes next to what it acts on.

## One boundary to keep

The camera raw decoder, `crates/lapstack-raw` with the patched `vendor/rawler`
inside it, is LGPL-2.1, and lapstack never links it: the command-line tool
loads it as a shared library and the browser app as a separate wasm module.
Changes to the decoder go in that crate, under that license; changes to the
rest of lapstack must not pull rawler into the engine. `vendor/rawler/
LAPSTACK-PATCH.md` explains the arrangement and how to move to a newer rawler.

A new dependency means regenerating `THIRD-PARTY.md` (`just third-party`)
and checking that the crate lands under a license lapstack can carry — the
script takes the first license it knows from the crate's SPDX expression and
otherwise the first one named, so an unfamiliar license shows up as its own
heading rather than as an error.

## Security

A security problem goes to security@ragtux.com rather than to a public
issue; see `SECURITY.md`.
