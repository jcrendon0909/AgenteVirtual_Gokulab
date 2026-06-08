# {
#   "object": "whatsapp_business_account",
#   "entry": [
#     {
#       "id": "<CUSTOMER_WABA_ID>",
#       "changes": [
#         {
#           "value": {
#             "messaging_product": "whatsapp",
#             "metadata": {
#               "display_phone_number": "<CUSTOMER_DISPLAY_PHONE_NUMBER>",
#               "phone_number_id": "<CUSTOMER_PHONE_NUMBER_ID>"
#             },
#             "history": [
#               {
#                 "metadata": {
#                   "phase": <PHASE>,
#                   "chunk_order": <CHUNK_ORDER>,
#                   "progress": <PROGRESS>
#                 },
#                 "threads": [
#                   /* First chat history thread object */
#                   {
#                     "id": "<WHATSAPP_USER_PHONE_NUMBER>",           <!-- CHANGED -->
#                     "context": {                                    <!-- ADDED -->
#                       "wa_id": "<WHATSAPP_USER_PHONE_NUMBER>",      <!-- ADDED -->
#                       "user_id": "<BSUID>",                         <!-- ADDED -->
#
#                       <!-- Only included if parent BSUIDs enabled before sync request -->
#                       "parent_user_id": "<PARENT_BSUID>",           <!-- ADDED -->
#
#                       <!-- Only included if user has enabled usernames feature before sync request -->
#                       "username": "<USERNAME>"                      <!-- ADDED -->
#
#                     },
#                     "messages": [
#                       /* First message object in thread */
#                       {
#                         "from": "<BUSINESS_OR_WHATSAPP_USER_PHONE_NUMBER>",  <!-- CHANGED -->
#                         "from_user_id" : "<BSUID>",                 <!-- ADDED -->
#
#                         <!-- Only included if parent BSUIDs enabled before sync request -->
#                         "from_parent_user_id": "<PARENT_BSUID>",    <!-- ADDED -->
#
#                         "to": "<WHATSAPP_USER_PHONE_NUMBER>",
#                         "id": "<WHATSAPP_MESSAGE_ID>",
#                         "timestamp": "<DEVICE_TIMESTAMP>,
#                         "type": "<MESSAGE_TYPE>",
#                         "<MESSAGE_TYPE>": {
#                           <MESSAGE_CONTENTS>
#                         },
#                         "history_context": {
#                           "status": "<MESSAGE_STATUS>"
#                         }
#                       },
#                       /* Additional message objects in thread would follow, if any */
#                     ]
#                   },
#                   /* Additional chat history thread objects would follow, if any */
#                 ]
#               }
#             ]
#           },
#           "field": "history"
#         }
#       ]
#     }
#   ]
# }

from .message import Message


class HistoryMessage: ...


class HistoryMediaMessage(Message):
    """
    Attributes:
        id: The message ID (If you want to reply to the message, use ``message_id_to_reply`` instead).
        metadata: The metadata of the message (which phone number was sent from).
        type: The message type (See :class:`MessageType`).
        to_user: The recipient of the message.
        chat: The chat where the message was sent to (private or group).
        timestamp: The timestamp when the message was sent (in UTC).
        reply_to_message: The message to which this message is a reply (if any).
        forwarded: Whether the message was forwarded.
        forwarded_many_times: Whether the message was forwarded more than 5 times. (when ``True``, ``forwarded`` will be ``True`` as well)
        text: The text of the message.
        image: The image of the message.
        video: The video of the message.
        sticker: The sticker of the message.
        document: The document of the message.
        audio: The audio of the message.
        voice: The voice note of the message (shorthand for ``audio`` if it's a voice note).
        caption: The caption of the message media (Optional, only available for image video and document messages).
        reaction: The reaction of the message.
        location: The location of the message.
        contacts: The contacts of the message.
        order: The order of the message.
        unsupported: The unsupported content of the message.
        error: The error of the message.
        shared_data: Shared data between handlers.
    """

    _webhook_field = "history"
    _messages_fields = (
        "messages",
        "message_echoes",
    )
