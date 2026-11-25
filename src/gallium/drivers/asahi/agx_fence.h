/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates.
 * Copyright 2018-2019 Alyssa Rosenzweig
 * Copyright 2018-2019 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "pipe/p_state.h"
#include "util/u_queue.h"

struct agx_context;
struct tc_unflushed_batch_token;

struct pipe_fence_handle {
   struct pipe_reference reference;
   struct util_queue_fence ready;
   struct tc_unflushed_batch_token *tc_token;
   uint32_t syncobj;
   bool signaled;
   bool faulted;
};

void agx_fence_reference(struct pipe_screen *pscreen,
                         struct pipe_fence_handle **ptr,
                         struct pipe_fence_handle *fence);

bool agx_fence_finish(struct pipe_screen *pscreen, struct pipe_context *ctx,
                      struct pipe_fence_handle *fence, uint64_t timeout);

int agx_fence_get_fd(struct pipe_screen *screen, struct pipe_fence_handle *f);

struct pipe_fence_handle *agx_fence_from_fd(struct agx_context *ctx, int fd,
                                            enum pipe_fd_type type);

struct pipe_fence_handle *agx_fence_create(struct agx_context *ctx);

struct pipe_fence_handle *
agx_fence_create_unflushed(struct pipe_context *pctx,
                           struct tc_unflushed_batch_token *token);

void agx_fence_populate(struct agx_context *ctx, struct pipe_fence_handle *f);

void agx_create_fence_fd(struct pipe_context *pctx,
                         struct pipe_fence_handle **pfence, int fd,
                         enum pipe_fd_type type);

void agx_fence_server_sync(struct pipe_context *pctx,
                           struct pipe_fence_handle *f, uint64_t value);
