import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Index from "./pages/Index";
import ClusterExplorer from "./pages/ClusterExplorer";
import NewsChatPage from "./pages/NewsChatPage";
import ScenarioPanel from "./pages/ScenarioPanel";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<NewsChatPage />} />
          <Route path="/news-chat" element={<Navigate to="/" replace />} />
          <Route path="/dashboard" element={<Index />} />
          <Route path="/clusters" element={<ClusterExplorer />} />
          <Route path="/scenarios" element={<ScenarioPanel />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
