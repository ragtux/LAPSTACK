# Reporting a security problem

Write to **security@ragtux.com** with what you found and how to reproduce it.
You will get a reply from the person who wrote the code, a fix, and credit if
you want it. Please give us a reasonable time to fix a problem before
publishing it.

lapstack runs on your own machine — in a browser tab, as a desktop
application, or as a command-line tool — and makes no network requests of its
own, so the surface is the files it reads and writes. A crafted image or raw
file that crashes it or makes it misbehave is the kind of report this address
is for; so is anything on lapstack.ragtux.com or lapstack.app.ragtux.com.

`/.well-known/security.txt` on both hosts says the same.
