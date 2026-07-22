"use client";

import { useRef, useState } from "react";
import { api, errorMessage, type ConsentBasis, type LeadImportResult } from "@/lib/api";
import { Button } from "./ui/Button";
import { Field, fieldControlClass } from "./ui/Field";
import { Table, Td, Th } from "./ui/Table";

type Props = {
  /** Absent while a campaign is still being created — the parent uploads the
   * chosen file itself once it has an id. */
  campaignId?: string;
  onImported?: () => void;
  /** Hand the chosen file/consent up instead of uploading here, for the
   * create form where the campaign doesn't exist yet. */
  onChange?: (file: File | null, consentBasis?: ConsentBasis) => void;
};

const ACCEPT = ".csv,.xlsx,.xlsm,.txt";

/** Bulk lead import. Deliberately makes the consent question explicit and
 * unticked by default: uploading a spreadsheet is not consent, and stamping
 * "explicit consent, just now" onto 200 numbers because a checkbox happened to
 * default on is exactly how a compliance problem gets created. Leads without
 * it still import — they just never dial, and the result says how many. */
export function LeadsUpload({ campaignId, onImported, onChange }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [affirmed, setAffirmed] = useState(false);
  const [result, setResult] = useState<LeadImportResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const consentBasis: ConsentBasis | undefined = affirmed ? "explicit" : undefined;

  function pick(next: File | null, nextAffirmed = affirmed) {
    setFile(next);
    setResult(null);
    setError(null);
    onChange?.(next, nextAffirmed ? "explicit" : undefined);
  }

  async function handleUpload() {
    if (!file || !campaignId) return;
    setUploading(true);
    setError(null);
    try {
      const res = await api.uploadLeads(campaignId, file, consentBasis);
      setResult(res);
      // Clear the picker so the same file isn't uploaded twice by reflex —
      // the import is idempotent, but a second run looks like it did nothing.
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
      onImported?.();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <Field
        label="Leads file"
        hint="CSV or Excel with a name and a phone column — headers like Name and Phone / Mobile / Contact / WhatsApp Number are found automatically. Leads are called in the order the rows appear."
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className={`${fieldControlClass} file:mr-3 file:rounded file:border-0 file:bg-neutral-100 file:px-3 file:py-1 file:text-sm dark:file:bg-neutral-800 dark:file:text-neutral-200`}
          onChange={(e) => pick(e.target.files?.[0] ?? null)}
        />
      </Field>

      <label className="flex items-start gap-2 text-sm text-neutral-700 dark:text-neutral-300">
        <input
          type="checkbox"
          className="mt-1"
          checked={affirmed}
          onChange={(e) => {
            setAffirmed(e.target.checked);
            onChange?.(file, e.target.checked ? "explicit" : undefined);
          }}
        />
        <span>
          I have explicit consent to call every number in this file.
          <span className="block text-xs text-neutral-500 dark:text-neutral-400">
            Applied only to rows with no consent column of their own. Leave it
            unticked and the leads still import — they just won&apos;t be dialled
            until consent is recorded.
          </span>
        </span>
      </label>

      {campaignId && (
        <Button onClick={handleUpload} disabled={!file || uploading} className="self-start">
          {uploading ? "Importing…" : "Import leads"}
        </Button>
      )}

      {error && <p className="text-sm text-status-critical">{error}</p>}
      {result && <ImportSummary result={result} />}
    </div>
  );
}

export function ImportSummary({ result }: { result: LeadImportResult }) {
  return (
    <div className="flex flex-col gap-2 rounded-md border border-neutral-200 p-3 dark:border-neutral-800">
      <p className="text-sm">
        <strong>{result.imported}</strong> of {result.received} row(s) imported
        {result.skipped > 0 && <> · <strong>{result.skipped}</strong> skipped</>}
        {" · "}
        <strong>{result.dialable}</strong> dial-eligible
      </p>
      {result.imported > 0 && result.dialable === 0 && (
        <p className="text-sm text-status-warning">
          None of these can be dialled yet — no consent was recorded for them.
        </p>
      )}
      {result.errors.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>Row</Th>
              <Th>Why it was skipped</Th>
            </tr>
          </thead>
          <tbody>
            {result.errors.map((e) => (
              <tr key={e.row_number}>
                <Td className="font-mono text-xs">{e.row_number}</Td>
                <Td>{e.reason}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </div>
  );
}
