import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { ModelFactory, OptimizationService, ModelInfo } from "@/lib/optimization";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2 } from "lucide-react";

type AcceptedFileType = ".pt" | ".onnx" | ".tflite";

const acceptedFileTypes: AcceptedFileType[] = [".pt", ".onnx", ".tflite"];

const ModelUploader = () => {
  const { toast } = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  
  // Get optimization service instance
  const optimizationService = OptimizationService.getInstance();

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = async (file: File) => {
    const fileExtension = `.${file.name.split('.').pop()}` as AcceptedFileType;
    
    if (!acceptedFileTypes.includes(fileExtension)) {
      toast({
        title: "Invalid file type",
        description: `Please upload a file with one of these extensions: ${acceptedFileTypes.join(", ")}`,
        variant: "destructive",
      });
      return;
    }
    
    setIsLoading(true);
    setFile(file);
    
    try {
      // Use the optimization service to load the model
      const info = await optimizationService.loadModelFile(file);
      setModelInfo(info);
      
      toast({
        title: "Model loaded successfully",
        description: `${file.name} has been loaded and is ready for optimization.`,
      });
    } catch (error) {
      toast({
        title: "Error loading model",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const loadPretrainedModel = async (modelId: string) => {
    setIsLoading(true);
    
    try {
      // Get model info to display
      const pretrainedModel = ModelFactory.getModelById(modelId);
      if (!pretrainedModel) {
        throw new Error(`Model ${modelId} not found`);
      }
      
      // Use the optimization service to load the pretrained model
      const info = await optimizationService.loadPretrainedModel(modelId);
      setModelInfo(info);
      setFile(null); // Clear the file since we're using a pretrained model
      
      toast({
        title: "Pretrained model loaded",
        description: `${pretrainedModel.name} has been loaded and is ready for optimization.`,
      });
    } catch (error) {
      toast({
        title: "Error loading pretrained model",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const clearModel = () => {
    setFile(null);
    setModelInfo(null);
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Model Selection</CardTitle>
        <CardDescription>
          Upload your own model or select from our pretrained models
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="upload" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="upload">Upload Your Model</TabsTrigger>
            <TabsTrigger value="pretrained">Pretrained Models</TabsTrigger>
          </TabsList>
          
          <TabsContent value="upload" className="pt-4">
            {!file && !modelInfo ? (
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragging
                    ? "border-primary bg-primary/5"
                    : "border-muted-foreground/20 hover:border-primary/50"
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById("file-upload")?.click()}
              >
                <div className="flex flex-col items-center justify-center gap-2">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="40"
                    height="40"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-muted-foreground"
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>

                  <div className="mt-2">
                    <p className="text-lg font-medium">
                      Drag and drop your model file here
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      or click to browse your files
                    </p>
                  </div>
                </div>
                <input
                  id="file-upload"
                  type="file"
                  accept=".pt,.onnx,.tflite"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </div>
            ) : (
              <div className="bg-muted/50 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="bg-primary/10 p-2 rounded-md">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="text-primary"
                      >
                        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                        <polyline points="14 2 14 8 20 8" />
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium">{file ? file.name : modelInfo?.name}</p>
                      {file && (
                        <p className="text-xs text-muted-foreground">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      )}
                      {modelInfo && !file && (
                        <p className="text-xs text-muted-foreground">
                          {modelInfo.architecture} model
                        </p>
                      )}
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearModel}
                    disabled={isLoading}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="text-muted-foreground"
                    >
                      <line x1="18" y1="6" x2="6" y2="18"></line>
                      <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                    <span className="sr-only">Remove model</span>
                  </Button>
                </div>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="pretrained" className="pt-4">
            <div className="grid grid-cols-1 gap-4">
              {isLoading ? (
                <div className="flex items-center justify-center p-8">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  <span className="ml-2">Loading model...</span>
                </div>
              ) : (
                ModelFactory.PRETRAINED_MODELS.map((model) => (
                  <div
                    key={model.id}
                    className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 cursor-pointer"
                    onClick={() => loadPretrainedModel(model.id)}
                  >
                    <div>
                      <h4 className="font-medium">{model.name}</h4>
                      <p className="text-sm text-muted-foreground">
                        {model.description}
                      </p>
                      <div className="flex gap-3 mt-1">
                        <span className="text-xs bg-primary/10 text-primary rounded-full px-2 py-0.5">
                          {model.size} MB
                        </span>
                        <span className="text-xs bg-primary/10 text-primary rounded-full px-2 py-0.5">
                          {model.accuracy}% mAP
                        </span>
                        <span className="text-xs bg-primary/10 text-primary rounded-full px-2 py-0.5">
                          {model.inferenceTime} ms
                        </span>
                      </div>
                    </div>
                    <Button variant="outline" size="sm">
                      Select
                    </Button>
                  </div>
                ))
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter>
        <p className="text-xs text-muted-foreground">
          Your model will be processed locally in your browser.
          No data is stored on our servers longer than necessary for optimization.
        </p>
      </CardFooter>
    </Card>
  );
};

export default ModelUploader;
