{{- define "ai-residency.labels" -}}
app.kubernetes.io/part-of: ai-residency
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{/*
The pod- and container-level security contexts, defined once so backends, the frontend,
Redis, the init containers and the training Jobs cannot drift apart. Each takes the root
context as its argument.
*/}}
{{- define "ai-residency.podSecurityContext" -}}
{{- if .Values.securityContext.enabled }}
securityContext:
  {{- toYaml .Values.securityContext.pod | nindent 2 }}
{{- end }}
{{- end -}}

{{- define "ai-residency.containerSecurityContext" -}}
{{- if .Values.securityContext.enabled }}
securityContext:
  {{- toYaml .Values.securityContext.container | nindent 2 }}
{{- end }}
{{- end -}}

{{/*
Resource requests and limits for one container. Takes a dict of {ctx, spec}, where spec
is a backend entry or the frontend/redis block.

No CPU limit is emitted anywhere, deliberately: a CPU limit throttles rather than kills,
and throttling a model mid-inference makes a fast request slow for no benefit here.
*/}}
{{- define "ai-residency.resources" -}}
{{- $spec := .spec -}}
{{- $cpu := $spec.cpuRequest | default (.defaultCpuRequest | default "") -}}
{{- if or $cpu $spec.memoryRequest $spec.memoryLimit }}
resources:
  {{- if or $cpu $spec.memoryRequest }}
  requests:
    {{- if $cpu }}
    cpu: {{ $cpu }}
    {{- end }}
    {{- if $spec.memoryRequest }}
    memory: {{ $spec.memoryRequest }}
    {{- end }}
  {{- end }}
  {{- if $spec.memoryLimit }}
  limits:
    memory: {{ $spec.memoryLimit }}
  {{- end }}
{{- end }}
{{- end -}}

{{/*
One entry of a pod's `volumes:` list. Takes {ctx, vol}.

A volume with `kind: hfCache` is resolved through .Values.hfCache, so the single-node
hostPath default can be swapped for a PersistentVolumeClaim -- or dropped entirely --
without editing every service that mounts the cache. Anything else is taken literally.
*/}}
{{- define "ai-residency.volumeSource" -}}
{{- $ctx := .ctx -}}
{{- $vol := .vol -}}
{{- if eq ($vol.kind | default "") "hfCache" -}}
{{- if eq $ctx.Values.hfCache.mode "pvc" -}}
persistentVolumeClaim:
  claimName: {{ required "hfCache.claimName is required when hfCache.mode is pvc" $ctx.Values.hfCache.claimName }}
{{- else if eq $ctx.Values.hfCache.mode "hostPath" -}}
hostPath:
  path: {{ required "hfCache.hostPath is required when hfCache.mode is hostPath" (tpl $ctx.Values.hfCache.hostPath $ctx) }}
  type: DirectoryOrCreate
{{- else -}}
emptyDir: {}
{{- end }}
{{- else -}}
hostPath:
  path: {{ tpl $vol.hostPath $ctx }}
  type: {{ $vol.type }}
{{- end }}
{{- end -}}

{{/*
The repository an image lives in, derived unless a service names one.

Every image this repo builds is `<registry>/<chart>-<service>-serve` or `-train`, with
the frontend the one exception (no suffix). That rule was written out eighteen times in
values.yaml, once per backend per image kind, and the key it derives from was sitting
right there in the `range`. The drift it allowed was quiet: rename a backend key and
the Deployment, the Service and the frontend's env var all follow, while `image:` goes
on naming the old repository -- a release that comes up running the previous service
under the new name.

An explicit `image:` still wins, for a service that does not follow the convention.

{{ include "ai-residency.repository" (dict "spec" $svc "ctx" $ "name" $name "suffix" "-serve") }}
*/}}

`deploy/images.yaml` is a values file naming every image by digest, merged over these
defaults with `helm upgrade -f ../images.yaml`. A digest cannot be moved, and it is
per-image -- which a single shared `image.tag` could not be.

There is no flag to turn that off, and no need for one: supplying the overlay pins the
release, omitting it falls back to `image.tag`. That fallback is what a local cluster
wants, since images built on this machine have their own digests and would never match
a published one. This used to read `image.useDigest` as well, which was a second switch
for the same decision the presence of a digest already makes.

Takes a dict: `spec` (the service's own values) and `ctx` (the root context).
*/}}
{{- define "ai-residency.repository" -}}
{{- $ctx := .ctx -}}
{{- $spec := .spec -}}
{{- if $spec.image -}}
{{ $spec.image }}
{{- else -}}
{{ $ctx.Values.image.registry }}/{{ $ctx.Chart.Name }}-{{ .name }}{{ .suffix }}
{{- end -}}
{{- end -}}

{{- define "ai-residency.image" -}}
{{- $spec := .spec -}}
{{- $ctx := .ctx -}}
{{- $repo := include "ai-residency.repository" (dict "spec" $spec "ctx" $ctx "name" .name "suffix" (.suffix | default "")) -}}
{{- if $spec.digest -}}
{{ $repo }}@{{ $spec.digest }}
{{- else -}}
{{ $repo }}:{{ $ctx.Values.image.tag }}
{{- end -}}
{{- end -}}
