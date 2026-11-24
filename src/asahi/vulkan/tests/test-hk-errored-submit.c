/* SPDX-License-Identifier: MIT
 * A command buffer whose recording failed must never reach the GPU:
 * hk_queue_submit() refuses the submission when
 * hk_submit_first_errored_cmd_buffer() finds one. No GPU used.
 */
#include <stdio.h>
#include <stdlib.h>
#include "hk_queue.h"
#include "vk_command_buffer.h"
#include "vk_queue.h"

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "line %d: %s\n", __LINE__, #x);                       \
         abort();                                                              \
      }                                                                        \
   } while (0)

int
main(void)
{
   struct vk_command_buffer cmd[3] = {0};
   struct vk_command_buffer *list[3] = {&cmd[0], &cmd[1], &cmd[2]};
   struct vk_queue_submit submit = {
      .command_buffer_count = 3,
      .command_buffers = list,
   };

   /* Clean recordings, and an empty submission, pass. */
   CHECK(hk_submit_first_errored_cmd_buffer(&submit) == NULL);
   submit.command_buffer_count = 0;
   CHECK(hk_submit_first_errored_cmd_buffer(&submit) == NULL);
   submit.command_buffer_count = 3;

   /* vk_command_buffer_set_error() keeps the first error; the first
    * errored command buffer of the batch is reported. */
   cmd[2].record_result = VK_ERROR_FEATURE_NOT_PRESENT;
   CHECK(hk_submit_first_errored_cmd_buffer(&submit) == &cmd[2]);
   cmd[1].record_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
   CHECK(hk_submit_first_errored_cmd_buffer(&submit) == &cmd[1]);

   /* Only the command buffers of this submission are looked at. */
   submit.command_buffer_count = 1;
   CHECK(hk_submit_first_errored_cmd_buffer(&submit) == NULL);

   printf("hk_errored_submit: OK\n");
   return 0;
}
