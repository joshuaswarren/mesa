#!/usr/bin/env bash
# Aurora Silicon assistant-file policy guard.
#
# The canonical policy is .github/aurora/AGENTS.md on the default branch.
# AGENTS.md, CLAUDE.md, GEMINI.md and .github/copilot-instructions.md may
# exist only as byte-identical regular-file copies of it. On a branch that
# is not a policy branch, a copy of the branch's own .github/aurora/AGENTS.md
# is also accepted: that branch forked from an older default branch and
# merging it brings the copies forward with the policy file. A copy path
# that holds anything else is rewritten to the policy file, never deleted,
# so a heal commit cannot conflict with the default branch. Every other
# assistant instruction file (any directory depth) is a violation, except
# paths listed one per line in .github/aurora/agent-policy-allow on the
# default branch.
#
# This script works only with paths, modes and object ids. It never reads
# the contents of an assistant instruction file and never checks one out.
#
# Only policy branches are ever committed to. On any other branch (feature
# and pull request branches, mirrors) push mode reports what it finds and
# changes nothing: a commit there would move a pull request head under its
# author and reviewers, and it would carry no human sign-off. The pull
# request check below gates what such a branch may merge, and the merge
# result takes the copies from the default branch unless the branch itself
# changed them. sweep visits the policy branches only.
#
# In pr mode the check runs on the merge result when GitHub has one
# (refs/pull/N/merge), else on the head. There the policy file must exist
# and every copy must exist and equal the policy file on the default
# branch. A pull request that changes the policy file itself passes only
# with POLICY_CHANGE_OK=1 (the workflow sets it from a maintainer-applied
# label), and then its copies may also equal its own policy file.
#
# pr-text mode checks the pull request's text, not its files: the body has
# the two lines of agent-guidelines policy/pr-structure.md that CI enforces
# ('Tested:' and 'Risk:' with the door and the Macs), and the title, body
# and the author's own commit messages carry no AI attribution lines. Pull requests created before STRUCTURE_SINCE, and drafts, get the
# findings as warnings only.
#
# Usage: agent-policy.sh push <branch> | sweep | pr <ref-or-sha> [<fallback-ref>] | pr-text
#
# Environment:
#   DEFAULT_BRANCH   default branch name (required)
#   POLICY_BRANCHES  branches that carry the copies (default: DEFAULT_BRANCH)
#   DRY_RUN=1        print what would happen, change nothing
#   LOCAL=1          operate on local refs/heads/* (bare clone testing),
#                    no fetch and no push
#   REMOTE           remote name (default: origin)
#   POLICY_CHANGE_OK=1  pr mode: the pull request may change the policy file
#   PR_TITLE, PR_BODY   pr-text mode: the pull request's title and body
#   PR_CREATED          pr-text mode: creation time (ISO 8601)
#   PR_DRAFT=true       pr-text mode: draft, warn only
#   STRUCTURE_SINCE     pr-text mode: enforce on PRs created at or after this
#                       time (default 2026-11-25T00:00:00Z)
#   COMMITS_FILE        pr-text mode: the author's commit messages, each
#                       starting with a line "commit <sha>"
set -u -o pipefail

MODE=${1:-}
ARG=${2:-}
REMOTE=${REMOTE:-origin}
DRY_RUN=${DRY_RUN:-0}
LOCAL=${LOCAL:-0}
[ "$MODE" = pr-text ] || : "${DEFAULT_BRANCH:?DEFAULT_BRANCH must be set}"
POLICY_BRANCHES=${POLICY_BRANCHES:-${DEFAULT_BRANCH:-}}

CANON_PATH=.github/aurora/AGENTS.md
ALLOW_PATH=.github/aurora/agent-policy-allow
COPIES=(AGENTS.md CLAUDE.md GEMINI.md .github/copilot-instructions.md)
FILE_RE='^((agents?|claude|gemini|codex|copilot-instructions)\.md|\.(cursorrules|clinerules|windsurfrules))$'
DIR_RE='^\.(claude|gemini|cursor|codex)$'
BOT_NAME='github-actions[bot]'
BOT_EMAIL='41898282+github-actions[bot]@users.noreply.github.com'
ZERO=0000000000000000000000000000000000000000

log() { printf '%s\n' "$*" >&2; }

if [ "$LOCAL" = 1 ]; then NS=refs/heads; else NS=refs/remotes/$REMOTE; fi

fetch_heads() {
	[ "$LOCAL" = 1 ] && return 0
	# Tips only, trees only. Blobs are fetched lazily and only for our
	# own policy files.
	git fetch -q --no-tags --prune --depth=1 --filter=blob:none "$REMOTE" \
		"+refs/heads/$1:$NS/$1"
}

is_policy_branch() {
	local b
	for b in $POLICY_BRANCHES; do [ "$b" = "$1" ] && return 0; done
	return 1
}

CANON=
ALLOW=()
load_canonical() {
	CANON=$(git rev-parse -q --verify "$NS/$DEFAULT_BRANCH:$CANON_PATH" 2>/dev/null || true)
	if [ -n "$CANON" ] && [ "$(git cat-file -t "$CANON")" != blob ]; then CANON=; fi
	local allow
	allow=$(git rev-parse -q --verify "$NS/$DEFAULT_BRANCH:$ALLOW_PATH" 2>/dev/null || true)
	ALLOW=()
	if [ -n "$allow" ]; then
		while IFS= read -r line; do
			line=${line%%#*}; line=${line%"${line##*[![:space:]]}"}
			[ -n "$line" ] && ALLOW+=("$line")
		done < <(git cat-file blob "$allow")
	fi
	log "canonical $CANON_PATH on $DEFAULT_BRANCH: ${CANON:-<missing>}"
}

is_copy_path() {
	local c
	for c in "${COPIES[@]}"; do [ "$c" = "$1" ] && return 0; done
	return 1
}

is_allowed() {
	local a
	for a in "${ALLOW[@]+"${ALLOW[@]}"}"; do [ "$a" = "$1" ] && return 0; done
	return 1
}

# scan_commit <commit> <expected-oid> <trust-own>
# Print "mode type oid<TAB>path" for each violation in the commit: copies
# must be regular files with blob <expected-oid>. With <trust-own> = 1 (a
# branch that is not a policy branch), a copy of the commit's own policy
# file is accepted too: it is a consistent older snapshot of the default
# branch. Print "missing<TAB>path" for each copy path that is absent.
scan_commit() {
	local commit=$1 expect=$2 trust=$3 meta mode type oid path c own=
	declare -A seen=()
	if [ "$trust" = 1 ]; then
		own=$(git rev-parse -q --verify "$commit:$CANON_PATH" 2>/dev/null || true)
	fi
	# One awk pass picks the candidates: assistant files at any depth and
	# assistant directories (whose contents are not listed separately).
	while IFS=$'\t' read -r meta path; do
		read -r mode type oid <<<"$meta"
		is_allowed "$path" && continue
		if [ "$type" != tree ]; then
			[ "$path" = "$CANON_PATH" ] && continue
			if is_copy_path "$path"; then
				seen[$path]=1
				if [ "$mode" = 100644 ] && { { [ -n "$expect" ] && [ "$oid" = "$expect" ]; } ||
					{ [ -n "$own" ] && [ "$oid" = "$own" ]; }; }; then
					continue
				fi
			fi
		fi
		printf '%s %s %s\t%s\n' "$mode" "$type" "$oid" "$path"
	done < <(git ls-tree -r -t --full-tree "$commit" | FRE=$FILE_RE DRE=$DIR_RE awk -F'\t' '
		skip != "" && index($2, skip "/") == 1 { next }
		{ skip = ""; n = split($2, parts, "/"); base = tolower(parts[n]); split($1, m, " ") }
		m[2] == "tree" { if (base ~ ENVIRON["DRE"]) { print; skip = $2 } next }
		base ~ ENVIRON["FRE"] { print }')
	for c in "${COPIES[@]}"; do
		[ -z "${seen[$c]:-}" ] && printf 'missing\t%s\n' "$c"
	done
	return 0
}

# heal <branch>: returns 0 when clean or healed, 1 when push failed.
heal() {
	local b=$1 tip policy=0 out viol=() missing=() line idx newtree commit msg p
	tip=$(git rev-parse -q --verify "$NS/$b^{commit}") || { log "$b: no such branch"; return 0; }
	is_policy_branch "$b" && policy=1
	out=$(scan_commit "$tip" "$CANON" $((1 - policy)))
	while IFS= read -r line; do
		[ -z "$line" ] && continue
		if [ "${line%%$'\t'*}" = missing ]; then missing+=("${line#*$'\t'}")
		else viol+=("$line"); fi
	done <<<"$out"
	if [ "$policy" = 1 ] && [ -n "$CANON" ]; then :; else missing=(); fi
	if [ ${#viol[@]} -eq 0 ] && [ ${#missing[@]} -eq 0 ]; then
		log "$b: clean"
		return 0
	fi
	for line in "${viol[@]+"${viol[@]}"}"; do log "$b: violation ${line}"; done
	for p in "${missing[@]+"${missing[@]}"}"; do log "$b: restore copy $p"; done
	if [ "$policy" = 0 ]; then
		log "$b: not a policy branch, report only (the pull request check gates the merge)"
		return 0
	fi
	[ "$DRY_RUN" = 1 ] && { log "$b: dry run, no commit"; return 0; }

	idx=$(mktemp)
	export GIT_INDEX_FILE=$idx
	git read-tree "$tip"
	for line in "${viol[@]+"${viol[@]}"}"; do
		p=${line#*$'\t'}
		if is_copy_path "$p" && [ "$(printf '%s' "$line" | cut -d' ' -f2)" != tree ]; then
			# Never delete a copy: sync it to the policy file on the
			# default branch. Without a policy file, leave it alone.
			[ -n "$CANON" ] && git update-index --add --cacheinfo "100644,$CANON,$p"
			continue
		fi
		if [ "$(printf '%s' "$line" | cut -d' ' -f2)" = tree ]; then
			git ls-tree -r -z --full-tree --name-only "$tip" -- "$p" | while IFS= read -r -d '' f; do
				printf '0 %s\t%s\n' "$ZERO" "$f"
			done | git update-index --index-info
		else
			printf '0 %s\t%s\n' "$ZERO" "$p" | git update-index --index-info
		fi
	done
	if [ "$policy" = 1 ] && [ -n "$CANON" ]; then
		for p in "${COPIES[@]}"; do
			git update-index --add --cacheinfo "100644,$CANON,$p"
		done
	fi
	newtree=$(git write-tree)
	unset GIT_INDEX_FILE
	rm -f "$idx"
	if [ "$newtree" = "$(git rev-parse "$tip^{tree}")" ]; then
		log "$b: nothing to change"
		return 0
	fi
	msg=$(printf 'agent-policy: sync assistant instruction files to %s\n\nAurora agent policy: remove assistant instruction files that are not\npolicy copies, and sync AGENTS.md, CLAUDE.md, GEMINI.md and\n.github/copilot-instructions.md to %s on %s.\n' "$CANON_PATH" "$CANON_PATH" "$DEFAULT_BRANCH")
	commit=$(GIT_AUTHOR_NAME=$BOT_NAME GIT_AUTHOR_EMAIL=$BOT_EMAIL \
		GIT_COMMITTER_NAME=$BOT_NAME GIT_COMMITTER_EMAIL=$BOT_EMAIL \
		git commit-tree "$newtree" -p "$tip" -m "$msg")
	log "$b: commit $commit"
	if [ "$LOCAL" = 1 ]; then
		git update-ref "refs/heads/$b" "$commit" "$tip"
		return 0
	fi
	# Plain push. If the branch moved, give up; the next run retries.
	if git push -q "$REMOTE" "$commit:refs/heads/$b"; then
		log "$b: pushed"
		return 0
	fi
	log "$b: push rejected, leaving it for the next run"
	return 1
}

list_branches() {
	if [ "$LOCAL" = 1 ]; then
		git for-each-ref --format='%(refname:strip=2)' refs/heads/
	else
		git ls-remote --heads "$REMOTE" | awk '{ sub("^refs/heads/", "", $2); print $2 }'
	fi
}

# ---- pr-text: PR structure and attribution -------------------------------
STRUCTURE_DOC=https://github.com/aurora-silicon/agent-guidelines/blob/main/policy/pr-structure.md
# Lines that attribute work to a model or tool (case-insensitive ERE).
# Trailers naming people are fine: only product names and AI-vendor or
# bot addresses count, so "Co-authored-by: Claude Dupont" passes.
AI_NAME='(claude (opus|sonnet|haiku|fable|code)|chatgpt|gpt-[0-9]|github copilot|copilot|openai codex|codex|gemini( [0-9]| code| cli)|cursor agent|devin-ai|@anthropic\.com|@openai\.com|noreply@anthropic|\[bot\]@)'
ATTRIB_RE="^[[:space:]>*_-]*(assisted-by[[:space:]]*:|(co-authored-by|co-developed-by|signed-off-by|reviewed-by|tested-by)[[:space:]]*:.*$AI_NAME|generated (with|by) .*(claude|codex|chatgpt|gpt|gemini|copilot|llm|cursor|an? ai\\b))|claude\\.ai/(code|chat|share)|chatgpt\\.com/(share|c)/|claude-session|🤖"

# strip_html_comments: stdin -> stdout without <!-- ... --> (multi-line).
strip_html_comments() {
	perl -0777 -pe 's/<!--.*?-->//gs'
}

# strip_code: stdin -> stdout without fenced code blocks and inline code,
# so quoting a trailer as an example is not an attribution line.
strip_code() {
	perl -0777 -pe 's/^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$//gms; s/`[^`\n]*`//g'
}

# pr_field <what|tested|risk|not yet>: stdin = body; print the field's text.
# A field starts at a "Label:" line (bold, list marker or heading allowed)
# or a "## Label" heading, and runs to the next field, heading or <details>.
# Folded <details> blocks are not part of the visible text and are skipped.
pr_field() {
	awk -v want="$1" '
	tolower($0) ~ /<details/ { fold++; on = 0 }
	tolower($0) ~ /<\/details>/ { if (fold > 0) fold--; next }
	fold > 0 { next }
	{
		l = $0; sub(/^[ \t]*/, "", l); head = (l ~ /^#/)
		sub(/^#+[ \t]*/, "", l); sub(/^[-*+][ \t]+/, "", l); gsub(/[*_]/, "", l)
		low = tolower(l); name = ""
		if (low ~ /^(what|tested|risk|not yet)[ \t]*:/) {
			name = low; sub(/[ \t]*:.*/, "", name); rest = l; sub(/^[^:]*:[ \t]*/, "", rest)
		} else if (head && low ~ /^(what|tested|risk|not yet)[ \t]*$/) {
			name = low; sub(/[ \t]*$/, "", name); rest = ""
		}
		if (name != "") { on = (name == want); if (on) print rest; next }
		if (head || low ~ /^<details/) { on = 0; next }
		if (on) print $0
	}'
}

pr_text() {
	local body title enforce=1 errs=() tested risk macs line f
	title=${PR_TITLE:-}
	body=$(printf '%s' "${PR_BODY:-}" | tr -d '\r' | strip_html_comments)
	if [ "${PR_DRAFT:-false}" = true ]; then enforce=0; fi
	if [ -n "${PR_CREATED:-}" ] && [[ "$PR_CREATED" < "${STRUCTURE_SINCE:-2026-11-25T00:00:00Z}" ]]; then
		enforce=0
	fi

	tested=$(printf '%s\n' "$body" | pr_field tested | tr -d '[:space:]')
	[ -n "$tested" ] || errs+=("Add a 'Tested:' line: the Macs you ran it on at the head SHA and the result, e.g. 'Tested: J413 M2 Air at abc1234: 0 failed units'. Docs or CI only: 'Tested: no runtime change'.")
	risk=$(printf '%s\n' "$body" | pr_field risk | tr '\n' ' ')
	if [ -z "$(printf '%s' "$risk" | tr -d '[:space:]')" ]; then
		errs+=("Add a 'Risk:' line: the door and the Macs it can affect, e.g. 'Risk: two-way · M4 Pro' (two-way = a plain revert undoes it).")
	else
		printf '%s' "$risk" | grep -q -i -E '(one|two)[ -]way' ||
			errs+=("Risk: say 'one-way' (a revert does not undo it) or 'two-way' (a plain revert undoes it), e.g. 'Risk: two-way · M4 Pro'.")
		macs=$(printf '%s' "$risk" | sed -E 's/(one|two)[ -]way( door)?//I; s/[^[:alnum:]]+//g')
		[ -n "$macs" ] ||
			errs+=("Risk: add which Macs it can affect after the door, e.g. 'two-way · M1, M2' or 'two-way · all Macs' or 'two-way · none (no runtime change)'.")
	fi

	# Attribution: title, body and the author's own commits. Carried
	# third-party commits are not in COMMITS_FILE.
	while IFS= read -r line; do
		errs+=("AI attribution in the title: '$line'. Remove it.")
	done < <(printf '%s\n' "$title" | grep -i -E "$ATTRIB_RE" || true)
	while IFS= read -r line; do
		errs+=("AI attribution in the PR body: '$line'. Delete this line.")
	done < <(printf '%s\n' "$body" | strip_code | grep -i -E "$ATTRIB_RE" || true)
	if [ -n "${COMMITS_FILE:-}" ] && [ -s "$COMMITS_FILE" ]; then
		while IFS= read -r line; do
			errs+=("AI attribution in commit ${line%%: *}: '${line#*: }'. Commits carry the human author and Signed-off-by only: amend it if the branch is yours alone, else recreate the commit without the line on a new branch and open a new PR (never force-push a shared branch).")
		done < <(awk '/^commit [0-9a-f]+$/ { c = substr($2, 1, 12); next } { print c ": " $0 }' "$COMMITS_FILE" |
			grep -i -E "^[0-9a-f]*: ${ATTRIB_RE#^}" || true)
	fi

	if [ ${#errs[@]} -eq 0 ]; then
		log "PR text follows $STRUCTURE_DOC"
		return 0
	fi
	if [ "$enforce" = 1 ]; then
		for f in "${errs[@]}"; do printf '::error title=PR text::%s\n' "$f"; done
		log "Edit the PR description; the check re-runs on every edit. Guide: $STRUCTURE_DOC"
		return 1
	fi
	for f in "${errs[@]}"; do printf '::warning title=PR text::%s\n' "$f"; done
	log "Warnings only (draft, or opened before ${STRUCTURE_SINCE:-2026-11-25T00:00:00Z}). Guide: $STRUCTURE_DOC"
	return 0
}

rc=0
case "$MODE" in
pr-text)
	pr_text || rc=1
	exit $rc
	;;
esac

case "$MODE" in
push)
	[ -n "$ARG" ] || { log "push needs a branch"; exit 2; }
	fetch_heads "$DEFAULT_BRANCH"
	[ "$ARG" != "$DEFAULT_BRANCH" ] && fetch_heads "$ARG"
	load_canonical
	heal "$ARG" || rc=1
	;;
sweep)
	# Policy branches only; every other branch is left to its owner and
	# to the pull request check.
	branches=$(list_branches)
	if [ "$LOCAL" != 1 ]; then
		for b in $POLICY_BRANCHES; do
			printf '%s\n' "$branches" | grep -qxF -- "$b" && fetch_heads "$b"
		done
	fi
	load_canonical
	for b in $POLICY_BRANCHES; do
		printf '%s\n' "$branches" | grep -qxF -- "$b" || { log "$b: no such branch"; continue; }
		heal "$b" || rc=1
	done
	;;
pr)
	[ -n "$ARG" ] || { log "pr needs a ref"; exit 2; }
	FALLBACK=${3:-}
	fetch_heads "$DEFAULT_BRANCH"
	if [ "$LOCAL" != 1 ]; then
		# The merge result is what lands; an unmergeable pull request has
		# none, so check its head then.
		if git fetch -q --no-tags --depth=1 --filter=blob:none "$REMOTE" "+$ARG:refs/aurora/pr-head" 2>/dev/null; then
			log "checking $ARG"
		elif [ -n "$FALLBACK" ] &&
			git fetch -q --no-tags --depth=1 --filter=blob:none "$REMOTE" "+$FALLBACK:refs/aurora/pr-head"; then
			log "no $ARG (the pull request does not merge cleanly), checking $FALLBACK"
			log "if this fails on files you did not touch, update the branch from $DEFAULT_BRANCH"
		else
			log "cannot fetch $ARG"; exit 2
		fi
		head=refs/aurora/pr-head
	else
		head=$ARG
	fi
	load_canonical
	[ -n "$CANON" ] || { log "no $CANON_PATH on $DEFAULT_BRANCH; nothing to check"; exit 0; }
	trust=0
	pr_canon=$(git rev-parse -q --verify "$head:$CANON_PATH" 2>/dev/null || true)
	if [ -z "$pr_canon" ]; then
		log "This pull request deletes $CANON_PATH."
		exit 1
	fi
	if [ "$pr_canon" != "$CANON" ]; then
		if [ "${POLICY_CHANGE_OK:-0}" != 1 ]; then
			log "This pull request changes $CANON_PATH. A maintainer must review"
			log "the policy change and label the pull request agent-policy-change."
			exit 1
		fi
		log "policy change approved by label; copies may match the new policy file"
		trust=1
	fi
	scan=$(scan_commit "$head" "$CANON" "$trust")
	out=$(printf '%s\n' "$scan" | awk -F'\t' 'NF && $1 != "missing"')
	if [ -n "$out" ]; then
		log "Assistant instruction files that are not Aurora policy copies:"
		log "$out"
		log "Remove them. Aurora's policy lives in $CANON_PATH on $DEFAULT_BRANCH."
		exit 1
	fi
	out=$(printf '%s\n' "$scan" | awk -F'\t' '$1 == "missing" { print $2 }')
	if [ -n "$out" ]; then
		log "This pull request deletes Aurora policy copies:"
		log "$out"
		log "Restore them from $DEFAULT_BRANCH, e.g. git checkout origin/$DEFAULT_BRANCH -- ${COPIES[*]}"
		exit 1
	fi
	log "PR is clean"
	;;
*)
	log "usage: $0 push <branch> | sweep | pr <ref> [<fallback-ref>] | pr-text"
	exit 2
	;;
esac
exit $rc
