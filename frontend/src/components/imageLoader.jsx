import React, { useRef, useState } from "react";
import { Button } from "./ui/button";
import { Table } from "lucide-react";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Dialog, DialogContent, DialogDescription,
  DialogFooter, DialogHeader, DialogTitle,
} from "./ui/dialog";
import Papa from "papaparse";

const ImageLoader = ({ onImagesLoaded, onError }) => {
  const csvRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [labelColumn, setLabelColumn] = useState("annotation");
  const [csvPreview, setCsvPreview] = useState(null);
  const [showDialog, setShowDialog] = useState(false);
  const [csvFile, setCsvFile] = useState(null);
  const [detectedLabels, setDetectedLabels] = useState([]);

  const LABEL_COLUMN_NAMES = [
    "annotation", "label", "class", "category", "target", "classification",
  ];

  const handleCsvSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.endsWith(".csv")) {
      onError("Please upload a CSV file.");
      e.target.value = "";
      return;
    }
    setIsLoading(true);
    setCsvFile(file);
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        Papa.parse(ev.target.result, {
          header: true,
          skipEmptyLines: true,
          complete: (result) => {
            const fatalErrors = result.errors.filter(
              (err) => err.type !== "Delimiter"
            );

            if (fatalErrors.length > 0) {
              onError(`Error parsing CSV: ${fatalErrors[0].message}`);
              setIsLoading(false);
              return;
            }
            const columns = result.meta.fields || [];
            let detected = null;
            for (const col of LABEL_COLUMN_NAMES) {
              if (columns.includes(col)) { detected = col; break; }
              const match = columns.find((c) => c.toLowerCase() === col);
              if (match) { detected = match; break; }
            }
            if (!detected && columns.length > 1) detected = columns[1];
            if (detected) setLabelColumn(detected);
            const unique = new Set();
            for (const row of result.data) {
              const val = detected && row[detected] ? row[detected].trim() : null;
              if (val) unique.add(val);
            }
            const allLabels = Array.from(unique).filter(Boolean).sort();
            setDetectedLabels(allLabels);
            setCsvPreview(result.data.slice(0, 5));
            if (allLabels.length > 0 && detected) {
              setShowDialog(true);
            } else {
              onImagesLoaded([file], "csv", ",");
            }
            setIsLoading(false);
          },
        });
      } catch (err) {
        onError(`Error reading CSV: ${err.message}`);
        setIsLoading(false);
      }
    };
    reader.onerror = () => { onError("Failed to read the CSV file."); setIsLoading(false); };
    reader.readAsText(file);
    e.target.value = "";
  };

  const handleConfirm = () => {
    if (detectedLabels.length > 0 && labelColumn) {
      onImagesLoaded([csvFile], "csv-with-labels", labelColumn, detectedLabels);
    } else {
      onImagesLoaded([csvFile], "csv", ",");
    }

    setShowDialog(false);
  };

  return (
    <>
      <div>
        <Button
          onClick={() => csvRef.current.click()}
          className="w-full"
          variant="default"
          disabled={isLoading}
        >
          <Table className="h-4 w-4 mr-2" />
          {isLoading ? "Processing..." : "Upload CSV with absolute image paths (labels optional)"}
        </Button>
        <input type="file" ref={csvRef} onChange={handleCsvSelect} accept=".csv" className="hidden" />
        <p className="text-xs text-gray-400 mt-1">
          CSV must have a <code>file_path</code> column with absolute paths (optional <code>annotation</code> column with class labels).
        </p>
      </div>

      <Dialog open={showDialog} onOpenChange={setShowDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Label Column</DialogTitle>
            <DialogDescription>
              Confirm which column contains the class labels. All labeled rows will be used for Episode 0 training.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4 space-y-3">
            <div>
              <Label htmlFor="label-column">Label Column</Label>
              <Input
                id="label-column"
                value={labelColumn}
                onChange={(e) => setLabelColumn(e.target.value)}
                className="mt-1"
                placeholder="e.g. annotation"
              />
            </div>
            {detectedLabels.length > 0 && (
              <p className="text-sm text-gray-600">
                Detected classes ({detectedLabels.length}):{" "}
                <span className="font-medium">{detectedLabels.join(", ")}</span>
              </p>
            )}
          </div>
          {csvPreview && csvPreview.length > 0 && (
            <div className="max-h-52 overflow-y-auto">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Preview (first 5 rows)</p>
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    {Object.keys(csvPreview[0]).map((col) => (
                      <th key={col} className={`px-3 py-2 text-left font-medium text-gray-500 ${col === labelColumn ? "bg-blue-100" : ""}`}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 bg-white">
                  {csvPreview.map((row, ri) => (
                    <tr key={ri}>
                      {Object.keys(row).map((col, ci) => (
                        <td key={`${ri}-${ci}`} className={`px-3 py-2 truncate max-w-[180px] ${col === labelColumn ? "bg-blue-50" : ""}`} title={row[col]}>{row[col]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <DialogFooter>
            <Button onClick={() => setShowDialog(false)} variant="outline">Cancel</Button>
            <Button onClick={handleConfirm}>Confirm & Upload</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default ImageLoader;