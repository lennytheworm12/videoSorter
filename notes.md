some notes about to project roadmap/plans

use playwright for python to scroll through the discord and grab the youtube links

first we open a headless:false session to login to discord and go to the right page(discord threads for each) scroll until we reach end of page
since the thread starts the at bottom maybe we scroll up(playwright specifics will sort out later)

read in each discord message extract the youtube link
and msg - link = title (ex gold top gnar) since its unknown whether the link will be start or end of msg

messages can be split into different forms:
1. desc, link
2. link, desc
3. link and or desc in the same message

in the third case the link and desc can be caught using the method above

but generally we assume that a message contains a youtube link scan the message for youtube link substring 
else if it has no youtube link we store in pending descriptions with timestamps
if an entries description is empty/weak try to fill from pending desc (have a check where pending_desc timestamp and video should be max 2 hours appart) (only try to fill if empty)

we'll have a sqlite database since our data is heavily keyed and ordered and we want to sort and filter the data
the schema for each data entry will look something like this:

entry {
video_id (primary key) (this is the part after v= or after youtu.be/ check for formatting of url)
video_url (required)
message_timestamp (required)
description
rank
champion
role (required)
}


