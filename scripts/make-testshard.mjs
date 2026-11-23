#!/usr/bin/env node
/**
 * Schreibt nach dem Build einen 30-MB-Test-Shard aus Zufallsbytes nach dist/diagnose/.
 * Zufallsbytes lassen sich nicht komprimieren – so misst #/diagnose die echte Transferrate.
 * Die Datei wird nicht eingecheckt.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const BYTES = 30 * 1024 * 1024; // muss zu src/pages/DiagnosePage.tsx passen
const dir = join(resolve(dirname(fileURLToPath(import.meta.url)), '..'), 'dist', 'diagnose');
mkdirSync(dir, { recursive: true });
writeFileSync(join(dir, 'testshard.bin'), randomBytes(BYTES));
console.log(`[testshard] ${BYTES / 1024 / 1024} MB → dist/diagnose/testshard.bin`);
