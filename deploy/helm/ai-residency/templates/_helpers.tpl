{{- define "ai-residency.labels" -}}
app.kubernetes.io/part-of: ai-residency
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
