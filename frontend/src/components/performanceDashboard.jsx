"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Alert, AlertDescription } from "../components/ui/alert";
import { AlertCircle, RefreshCw, TrendingUp } from "lucide-react";
import activeLearnAPI from "../services/activelearning";

const STRATEGY_LABELS = {
  least_confidence: "Least conf.",
  margin: "Margin",
  entropy: "Entropy",
  random: "Random",
};

const VAL_WARN_THRESHOLD = 0.05;

function fmtPct(v) {
  if (v == null) return "—";
  return `${Number(v).toFixed(1)}%`;
}

function strategyLabel(s) {
  return STRATEGY_LABELS[s] ?? s;
}

function MetricCard({ label, value, sub, color }) {
  const colorClass =
    color === "green"
      ? "text-emerald-600"
      : color === "blue"
        ? "text-blue-600"
        : "text-gray-800";
  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
        {label}
      </p>
      <p className={`text-2xl font-semibold font-mono ${colorClass}`}>
        {value}
      </p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-3 text-sm min-w-[180px]">
      <p className="font-medium text-gray-700 mb-2">{label}</p>
      {payload.map((p) => (
        <p
          key={p.name}
          className="flex justify-between gap-4"
          style={{ color: p.color }}
        >
          <span>{p.name}</span>
          <span className="font-mono">
            {fmtPct(p.value)}
          </span>
        </p>
      ))}
    </div>
  );
};

const PerformanceDashboard = ({ maxEpisodes = 10 }) => {
  const [episodes, setEpisodes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastFetched, setLastFetched] = useState(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await activeLearnAPI.getEpisodeHistory();
      const eps = (data.episodes ?? [])
        .sort((a, b) => a.episode - b.episode);
      setEpisodes(eps);
      setLastFetched(new Date());
    } catch (err) {
      setError(err.message ?? "Failed to load episode history");
    } finally {
      setLoading(false);
    }
  }, [maxEpisodes]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const interval = setInterval(load, episodes.length === 0 ? 5000 : 15000);
    return () => clearInterval(interval);
  }, [load, episodes.length]);

  const totalImages = episodes.length
    ? (episodes[0].labeled_size ?? 0) +
      (episodes[0].unlabeled_size ?? 0) +
      (episodes[0].validation_size ?? 0)
    : 0;

  const chartData = episodes.map((ep) => ({
    name: `Ep ${ep.episode}`,
    Accuracy:
      ep.best_val_acc != null ? parseFloat(ep.best_val_acc.toFixed(1)) : null,
    "F1 score": ep.f1_score != null ? parseFloat(ep.f1_score.toFixed(1)) : null,
    "Val. set %":
      ep.validation_size != null && totalImages > 0
        ? parseFloat(((ep.validation_size / totalImages) * 100).toFixed(1))
        : null,
  }));

  const accs = episodes.map((e) => e.best_val_acc ?? 0);
  const f1s = episodes.map((e) => e.f1_score ?? 0);
  const bestAcc = accs.length ? Math.max(...accs) : null;
  const latestAcc = accs.length ? accs[accs.length - 1] : null;
  const latestF1 = f1s.length ? f1s[f1s.length - 1] : null;
  const hasRealF1 = episodes.some((e) => e.f1_score != null);

  const latestValSize = episodes.length
    ? (episodes[episodes.length - 1].validation_size ?? 0)
    : 0;
  const valRatio = totalImages > 0 ? latestValSize / totalImages : 0;
  const valTooSmall = valRatio > 0 && valRatio < VAL_WARN_THRESHOLD;

  if (loading && episodes.length === 0) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-48">
          <div className="flex items-center gap-3 text-gray-400">
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span>Loading episode history…</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              <span>{error}</span>
              <Button
                variant="outline"
                size="sm"
                onClick={load}
                className="ml-4"
              >
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (episodes.length === 0) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center h-48 gap-3 text-gray-400">
          <TrendingUp className="h-8 w-8 opacity-30" />
          <p>No episodes recorded yet.</p>
          <p className="text-sm">
            Complete your first training episode to see metrics here.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <CardTitle className="text-base font-medium">
              Model performance 
            </CardTitle>
            <div className="flex items-center gap-3">
              {lastFetched && (
                <span className="text-xs text-gray-400">
                  Updated {lastFetched.toLocaleTimeString()}
                </span>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={load}
                disabled={loading}
                className="h-7 px-2 text-xs"
              >
                <RefreshCw
                  className={`h-3 w-3 mr-1 ${loading ? "animate-spin" : ""}`}
                />
                Refresh
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {valTooSmall && (
            <Alert className="border-amber-300 bg-amber-50">
              <AlertCircle className="h-4 w-4 text-amber-600" />
              <AlertDescription className="text-amber-800">
                The validation set is now only {latestValSize} images (
                {(valRatio * 100).toFixed(1)}% of total). Accuracy and F1
                estimates may be unreliable. You can stop here or continue.
              </AlertDescription>
            </Alert>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard
              label="Best accuracy"
              value={fmtPct(bestAcc)}
              color="green"
            />
            <MetricCard
              label="Latest accuracy"
              value={fmtPct(latestAcc)}
              sub={`episode ${episodes.length}`}
            />
            <MetricCard
              label="Latest F1 (weighted)"
              value={fmtPct(latestF1)}
              color="blue"
              sub={hasRealF1 ? "real" : "no data yet"}
            />
            <MetricCard
              label="Episodes done"
              value={episodes.length}              
            />
          </div>

          <div>
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
              Performance curve
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart
                data={chartData}
                margin={{ top: 4, right: 56, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11, fill: "#9ca3af" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  yAxisId="pct"
                  domain={[0, 100]}
                  tickFormatter={(v) => `${v}%`}
                  tick={{ fontSize: 11, fill: "#9ca3af" }}
                  axisLine={false}
                  tickLine={false}
                  width={42}
                />
                <YAxis
                  yAxisId="count"
                  orientation="right"
                  domain={[0, 100]}
                  tickFormatter={(v) => `${v}%`}
                  tick={{ fontSize: 11, fill: "#c4b5fd" }}
                  axisLine={false}
                  tickLine={false}
                  width={48}
                  label={{
                    value: "Val. %",
                    angle: 90,
                    position: "insideRight",
                    offset: 14,
                    fontSize: 10,
                    fill: "#c4b5fd",
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  iconType="circle"
                  iconSize={8}
                  wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
                />
                <ReferenceLine
                  yAxisId="pct"
                  y={80}
                  stroke="#d1d5db"
                  strokeDasharray="4 3"
                  label={{
                    value: "80%",
                    position: "insideTopRight",
                    fontSize: 10,
                    fill: "#9ca3af",
                  }}
                />
                <Line
                  yAxisId="pct"
                  type="monotone"
                  dataKey="Accuracy"
                  stroke="#059669"
                  strokeWidth={2}
                  dot={{ r: 4, fill: "#059669" }}
                  activeDot={{ r: 5 }}
                  connectNulls
                />
                <Line
                  yAxisId="pct"
                  type="monotone"
                  dataKey="F1 score"
                  stroke="#2563eb"
                  strokeWidth={2}
                  dot={{ r: 3.5, fill: "#2563eb" }}
                  activeDot={{ r: 5 }}
                  connectNulls
                />
                <Line
                  yAxisId="count"
                  type="monotone"
                  dataKey="Val. set %"
                  stroke="#a78bfa"
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  dot={{ r: 3, fill: "#a78bfa" }}
                  activeDot={{ r: 4 }}
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
            {!hasRealF1 && (
              <p className="text-xs text-amber-600 mt-1">
                F1 data will appear here after the next training episode
                completes.
              </p>
            )}
          </div>

          <div>
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
              Episode breakdown
            </p>
            <div className="rounded-lg border border-gray-200 overflow-hidden">
              <div className="overflow-x-auto">
                <table
                  className="w-full text-sm"
                  style={{ tableLayout: "fixed", minWidth: 600 }}
                >
                  <colgroup>
                    <col style={{ width: 60 }} />
                    <col style={{ width: 90 }} />
                    <col style={{ width: 100 }} />
                    <col style={{ width: 80 }} />
                    <col style={{ width: 80 }} />
                    <col style={{ width: 70 }} />
                    <col />
                  </colgroup>
                  <thead className="bg-gray-50 border-b border-gray-200">
                    <tr>
                      {[
                        "Ep.",
                        "Accuracy",
                        "F1 (weighted)",
                        "Labeled",
                        "Unlabeled",
                        "Val. %",
                        "Strategy",
                      ].map((h) => (
                        <th
                          key={h}
                          className="text-left px-3 py-2 text-xs font-medium text-gray-500 uppercase tracking-wide"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {episodes.slice(-10).map((ep) => {
                      const isBest =
                        ep.best_val_acc != null &&
                        Math.abs(ep.best_val_acc - (bestAcc ?? 0)) < 0.005;
                      const epValRatio =
                        totalImages > 0
                          ? (ep.validation_size ?? 0) / totalImages
                          : 0;
                      const epValSmall =
                        epValRatio > 0 && epValRatio < VAL_WARN_THRESHOLD;
                      return (
                        <tr
                          key={ep.episode}
                          className="hover:bg-gray-50 transition-colors"
                        >
                          <td className="px-3 py-2 font-mono text-xs text-gray-700">
                            {ep.episode}
                          </td>
                          <td className="px-3 py-2 font-mono text-xs font-medium text-gray-800">
                            {fmtPct(ep.best_val_acc)}
                          </td>
                          <td className="px-3 py-2 font-mono text-xs text-blue-700">
                            {fmtPct(ep.f1_score)}
                          </td>
                          <td className="px-3 py-2 font-mono text-xs text-gray-600">
                            {ep.labeled_size ?? "—"}
                          </td>
                          <td className="px-3 py-2 font-mono text-xs text-gray-600">
                            {ep.unlabeled_size ?? "—"}
                          </td>
                          <td className="px-3 py-2 font-mono text-xs">
                            <span
                              className={
                                epValSmall
                                  ? "text-amber-600 font-medium"
                                  : "text-gray-600"
                              }
                            >
                              {totalImages > 0 && ep.validation_size != null
                                ? `${((ep.validation_size / totalImages) * 100).toFixed(1)}%`
                                : "—"}
                              {epValSmall && " ⚠"}
                            </span>
                          </td>
                          <td className="px-3 py-2">
                            <span className="inline-block text-[10px] px-2 py-0.5 rounded-full bg-blue-50 text-blue-700">
                              {strategyLabel(ep.strategy)}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-2">
                Val. % = validation set size normalized by all input CSV records — ⚠ flags when it drops below 5%.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PerformanceDashboard;
