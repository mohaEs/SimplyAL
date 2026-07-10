import React, { useState, useRef } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "./ui/card";
import { Label } from "./ui/label";
import { Button } from "./ui/button";
import { Upload, CheckCircle, AlertCircle, FolderOpen } from "lucide-react";
import { Alert, AlertDescription } from "./ui/alert";
import activeLearnAPI from "../services/activelearning";

const ProjectImport = ({ onImportSuccess, onError }) => {
  const [isImporting, setIsImporting] = useState(false);
  const [importError, setImportError] = useState(null);
  const [importStatus, setImportStatus] = useState(null);
  const [imported, setImported] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [projectInfo, setProjectInfo] = useState(null);

  const projectFileRef = useRef(null);

  const handleProjectFileSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.endsWith(".zip")) {
      setImportError("Please select a ZIP file containing an exported project");
      return;
    }

    setSelectedFile(file);
    setImportError(null);
    setImportStatus(
      "Project file selected. Click 'Import Project' to continue."
    );
  };

  const handleImportProject = async () => {
    if (!selectedFile) {
      setImportError("Please select a project ZIP file");
      return;
    }

    try {
      setIsImporting(true);
      setImportError(null);
      setImportStatus("Importing project...");

      const result = await activeLearnAPI.importProject(selectedFile);

      console.log("Project import result:", result);
      setProjectInfo(result);
      setImportStatus(
        "Project imported successfully! Upload your images to continue training."
      );

      // Call the success callback with import result
      setImported(true);
      onImportSuccess({
        project_name: result.project_info.project_name,
        model_type: result.project_info.model_type || "resnet50",
        ...result,
      });
    } catch (error) {
      console.error("Import error:", error);
      setImportError(`Failed to import project: ${error.message}`);
      setImportStatus(null);
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FolderOpen className="h-5 w-5" />
          Import Exported Project
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Select Project ZIP File</Label>
          <div className="flex gap-2">
            <Button
              onClick={() => projectFileRef.current.click()}
              variant="outline"
              className="w-full"
              disabled={isImporting}
            >
              <Upload className="h-4 w-4 mr-2" />
              Select Project File (.zip)
            </Button>
            <input
              type="file"
              ref={projectFileRef}
              onChange={handleProjectFileSelect}
              className="hidden"
              accept=".zip"
            />
          </div>
          {selectedFile && (
            <div className="text-sm text-gray-600">
              Selected: {selectedFile.name}
            </div>
          )}
        </div>

        {projectInfo && (
          <Card className="bg-blue-50 border-blue-200">
            <CardContent className="pt-4">
              <h4 className="font-medium mb-2">Project Information</h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Project Name:</span>
                  <span className="font-medium">
                    {projectInfo.project_info.project_name}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Episodes Completed:</span>
                  <span>{projectInfo.project_info.current_episode}</span>
                </div>
                <div className="flex justify-between">
                  <span>Best Accuracy:</span>
                  <span>
                    {(
                      projectInfo.project_info.best_validation_accuracy || 0
                    ).toFixed(2)}
                    %
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Model Type:</span>
                  <span>{projectInfo.project_info.model_type}</span>
                </div>
                <div className="flex justify-between">
                  <span>Sampling Strategy:</span>
                  <span>{projectInfo.hyperparameters.sampling_strategy}</span>
                </div>
                <div className="flex justify-between">
                  <span>Batch Size:</span>
                  <span>{projectInfo.hyperparameters.batch_size}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {importError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{importError}</AlertDescription>
          </Alert>
        )}

        {importStatus && (
          <Alert>
            <CheckCircle className="h-4 w-4" />
            <AlertDescription>{importStatus}</AlertDescription>
          </Alert>
        )}
      </CardContent>
      <CardFooter className="flex flex-col gap-2">
        <Button
          className="w-full"
          onClick={handleImportProject}
          disabled={!selectedFile || isImporting || imported}
        >
          {isImporting ? (
            <span className="flex items-center gap-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              Importing Project...
            </span>
          ) : (
            "Import Project"
          )}
        </Button>
      </CardFooter>
    </Card>
  );
};

export default ProjectImport;
