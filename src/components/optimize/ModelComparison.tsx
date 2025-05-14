import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { OptimizationResult, SimulationService, HARDWARE_PROFILES } from "@/lib/optimization";

interface ModelComparisonProps {
  result: OptimizationResult;
}

const ModelComparison = ({ result }: ModelComparisonProps) => {
  // Calculate reduction percentages
  const sizeReduction = ((result.originalModel.size - result.optimizedModel.size) / result.originalModel.size) * 100;
  const speedup = ((result.originalModel.inferenceTime - result.optimizedModel.inferenceTime) / result.originalModel.inferenceTime) * 100;
  const accuracyChange = result.optimizedModel.accuracy - result.originalModel.accuracy;
  
  // Prepare data for charts
  const barChartData = [
    {
      name: "Size (MB)",
      Original: result.originalModel.size,
      Optimized: result.optimizedModel.size,
    },
    {
      name: "Inference Time (ms)",
      Original: result.originalModel.inferenceTime,
      Optimized: result.optimizedModel.inferenceTime,
    },
    {
      name: "Accuracy (%)",
      Original: result.originalModel.accuracy,
      Optimized: result.optimizedModel.accuracy,
    },
  ];
  
  // Radar chart data - normalized for better visualization
  const normalizeValue = (value: number, max: number) => (value / max) * 100;
  
  const radarChartData = [
    {
      metric: "Size",
      Original: normalizeValue(result.originalModel.size, Math.max(result.originalModel.size, result.optimizedModel.size)),
      Optimized: normalizeValue(result.optimizedModel.size, Math.max(result.originalModel.size, result.optimizedModel.size)),
    },
    {
      metric: "Speed",
      Original: normalizeValue(100 - result.originalModel.inferenceTime, 100),
      Optimized: normalizeValue(100 - result.optimizedModel.inferenceTime, 100),
    },
    {
      metric: "Accuracy",
      Original: normalizeValue(result.originalModel.accuracy, 100),
      Optimized: normalizeValue(result.optimizedModel.accuracy, 100),
    },
  ];
  
  // Simulate performance on different hardware profiles
  const jetsonPerformance = SimulationService.simulatePerformance(
    result.optimizedModel.size,
    result.optimizedModel.inferenceTime,
    HARDWARE_PROFILES["jetson-nano"]
  );
  
  const raspberryPiPerformance = SimulationService.simulatePerformance(
    result.optimizedModel.size,
    result.optimizedModel.inferenceTime,
    HARDWARE_PROFILES["raspberry-pi"]
  );
  
  const coralTPUPerformance = SimulationService.simulatePerformance(
    result.optimizedModel.size,
    result.optimizedModel.inferenceTime,
    HARDWARE_PROFILES["coral-tpu"]
  );
  
  // Hardware comparison data
  const hardwareComparisonData = [
    {
      name: "Jetson Nano",
      "FPS": 1000 / jetsonPerformance.inferenceTime,
      "Memory (%)": (jetsonPerformance.memoryUsage || 0) / HARDWARE_PROFILES["jetson-nano"].memory * 100,
    },
    {
      name: "Raspberry Pi",
      "FPS": 1000 / raspberryPiPerformance.inferenceTime,
      "Memory (%)": (raspberryPiPerformance.memoryUsage || 0) / HARDWARE_PROFILES["raspberry-pi"].memory * 100,
    },
    {
      name: "Coral TPU",
      "FPS": 1000 / coralTPUPerformance.inferenceTime,
      "Memory (%)": (coralTPUPerformance.memoryUsage || 0) / HARDWARE_PROFILES["coral-tpu"].memory * 100,
    },
  ];
  
  const handleDownload = () => {
    // Create a JSON representation of the result
    const resultBlob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(resultBlob);
    
    // Create a link and trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = `optimized_model_report_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Optimization Results</CardTitle>
        <Button variant="outline" onClick={handleDownload}>
          <Download className="mr-2 h-4 w-4" /> Download Report
        </Button>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-muted/50 p-4 rounded-lg text-center">
            <p className="text-muted-foreground text-sm">Size Reduction</p>
            <p className={`text-3xl font-bold ${sizeReduction >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {sizeReduction.toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {result.optimizedModel.size.toFixed(1)} MB (was {result.originalModel.size.toFixed(1)} MB)
            </p>
          </div>
          
          <div className="bg-muted/50 p-4 rounded-lg text-center">
            <p className="text-muted-foreground text-sm">Speed Improvement</p>
            <p className={`text-3xl font-bold ${speedup >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {speedup.toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {result.optimizedModel.inferenceTime.toFixed(1)} ms (was {result.originalModel.inferenceTime.toFixed(1)} ms)
            </p>
          </div>
          
          <div className="bg-muted/50 p-4 rounded-lg text-center">
            <p className="text-muted-foreground text-sm">Accuracy Change</p>
            <p className={`text-3xl font-bold ${accuracyChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {accuracyChange > 0 ? '+' : ''}{accuracyChange.toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {result.optimizedModel.accuracy.toFixed(1)}% (was {result.originalModel.accuracy.toFixed(1)}%)
            </p>
          </div>
        </div>
        
        <div className="space-y-6">
          <Tabs defaultValue="metrics">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="metrics">Metrics Comparison</TabsTrigger>
              <TabsTrigger value="radar">Performance Profile</TabsTrigger>
              <TabsTrigger value="hardware">Hardware Simulation</TabsTrigger>
            </TabsList>
            
            <TabsContent value="metrics" className="pt-4">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={barChartData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="Original" fill="#8884d8" name="Original Model" />
                    <Bar dataKey="Optimized" fill="#82ca9d" name="Optimized Model" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div className="mt-4 p-4 bg-muted/30 rounded-lg">
                <h3 className="font-medium mb-2">Applied Optimizations</h3>
                <div className="grid grid-cols-2 gap-2">
                  {result.optimizationSteps.map((step, index) => (
                    <div key={index} className="flex items-center">
                      <div className="w-2 h-2 bg-primary rounded-full mr-2"></div>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="radar" className="pt-4">
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarChartData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} />
                    <Radar
                      name="Original Model"
                      dataKey="Original"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.6}
                    />
                    <Radar
                      name="Optimized Model"
                      dataKey="Optimized"
                      stroke="#82ca9d"
                      fill="#82ca9d"
                      fillOpacity={0.6}
                    />
                    <Legend />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
              
              <div className="mt-4 p-4 bg-muted/30 rounded-lg">
                <h3 className="font-medium mb-2">Optimization Trade-offs</h3>
                <p className="text-sm text-muted-foreground">
                  {sizeReduction > 0 && speedup > 0 && accuracyChange >= -2 && (
                    "This optimization achieved significant size and speed improvements with minimal accuracy impact."
                  )}
                  {sizeReduction > 0 && speedup > 0 && accuracyChange < -2 && (
                    "This optimization reduced size and improved speed, but with some accuracy trade-off."
                  )}
                  {sizeReduction < 0 || speedup < 0 && (
                    "This optimization had mixed results. Consider trying different optimization parameters."
                  )}
                </p>
              </div>
            </TabsContent>
            
            <TabsContent value="hardware" className="pt-4">
              <div className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={hardwareComparisonData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                    <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="left" dataKey="FPS" fill="#8884d8" name="FPS (higher is better)" />
                    <Bar yAxisId="right" dataKey="Memory (%)" fill="#82ca9d" name="Memory Usage %" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div className="mt-4 p-4 bg-muted/30 rounded-lg">
                <h3 className="font-medium mb-2">Real-time Detection Capability</h3>
                <div className="grid grid-cols-3 gap-4 mt-3">
                  {hardwareComparisonData.map((hw) => (
                    <div key={hw.name} className="text-center">
                      <p className="font-medium">{hw.name}</p>
                      <div className={`text-sm mt-1 ${hw.FPS >= 10 ? 'text-green-500' : 'text-amber-500'}`}>
                        {hw.FPS >= 10 ? '✓ Real-time' : '⚠ Near real-time'}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {hw.FPS.toFixed(1)} FPS
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelComparison;
