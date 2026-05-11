/**
 * hooks/useFileUpload.ts
 * Wraps the upload endpoint with per-file optimistic state.
 */
"use client";

import { useCallback, useMemo } from "react";

import { uploadFile } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { UploadedFile } from "@/types";

interface UseFileUploadResult {
  files:      UploadedFile[];
  upload:     (file: File) => Promise<UploadedFile | null>;
  removeFile: (fileId: string) => void;
}

export function useFileUpload(): UseFileUploadResult {
  const sessionId   = useAppStore((s) => s.activeSessionId);
  const filesMap    = useAppStore((s) => s.files);
  const appendFile  = useAppStore((s) => s.appendFile);
  const removeFile  = useAppStore((s) => s.removeFile);
  const files       = useMemo(
    () => (sessionId ? filesMap[sessionId] ?? [] : []),
    [sessionId, filesMap],
  );

  const upload = useCallback(async (file: File): Promise<UploadedFile | null> => {
    if (!sessionId) return null;

    const tempId = crypto.randomUUID();
    const optimistic: UploadedFile = {
      fileId:     tempId,
      filename:   file.name,
      size:       file.size,
      uploadedAt: new Date().toISOString(),
      uploading:  true,
    };
    appendFile(sessionId, optimistic);

    try {
      const result = await uploadFile(file, sessionId);
      // Replace the optimistic record by removing the temp id and appending the real one
      removeFile(sessionId, tempId);
      appendFile(sessionId, result);
      return result;
    } catch (err) {
      removeFile(sessionId, tempId);
      const failed: UploadedFile = {
        ...optimistic,
        uploading: false,
        error: err instanceof Error ? err.message : "Upload failed",
      };
      appendFile(sessionId, failed);
      return null;
    }
  }, [sessionId, appendFile, removeFile]);

  const remove = useCallback((fileId: string) => {
    if (sessionId) removeFile(sessionId, fileId);
  }, [sessionId, removeFile]);

  return { files, upload, removeFile: remove };
}
