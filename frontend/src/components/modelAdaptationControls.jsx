import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import activeLearnAPI from "@/services/activelearning";

const ModelAdaptationControls = ({ onAdaptSuccess, onError, disabled }) => {
  const [freezeLayers, setFreezeLayers] = useState(true);
  const [adaptationType, setAdaptationType] = useState("full_finetune");
  const [isLoading, setIsLoading] = useState(false);

  const handleAdaptModel = async () => {
    setIsLoading(true);
    try {
      const result = await activeLearnAPI.adaptPretrainedModel(
        freezeLayers,
        adaptationType
      );
      onAdaptSuccess && onAdaptSuccess(result);
    } catch (error) {
      onError && onError(error.message || "Failed to adapt model");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Model Adaptation</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="freeze-layers">Freeze Base Layers</Label>
            <Switch
              id="freeze-layers"
              checked={freezeLayers}
              onCheckedChange={setFreezeLayers}
              disabled={disabled}
            />
          </div>

          <div className="space-y-2">
            <Label>Adaptation Strategy</Label>
            <Select
              value={adaptationType}
              onValueChange={setAdaptationType}
              disabled={disabled}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select adaptation type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="full_finetune">Full Fine-tuning</SelectItem>
                <SelectItem value="last_layer">Last Layer Only</SelectItem>
                <SelectItem value="mid_layers">Mid + Last Layers</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="mt-2">
            <p className="text-sm text-gray-500 mb-2">
              {adaptationType === "full_finetune"
                ? "Fine-tune all layers (or only unfreezed ones). Best for similar domains."
                : adaptationType === "last_layer"
                ? "Only adapt classification layer. Best for very different domains."
                : "Adapt mid-level layers and classification layer. Balanced approach."}
            </p>
          </div>

          <Button
            className="w-full"
            onClick={handleAdaptModel}
            disabled={disabled || isLoading}
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Adapting Model...
              </span>
            ) : (
              "Adapt Model for Active Learning"
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelAdaptationControls;
