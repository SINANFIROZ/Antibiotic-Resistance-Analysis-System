export type MetricCard = {
  label: string;
  value: string;
  delta: string;
};

export type TrendDatum = {
  name: string;
  resistant: number;
  susceptible: number;
};

export type HeatmapDatum = {
  microbe: string;
  antibiotic: string;
  resistanceRate: number;
};
