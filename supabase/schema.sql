-- Supabase schema for hosted query-serving data.
-- Run in the Supabase SQL editor before running cloud.migrate_supabase.

create extension if not exists vector;

create table if not exists public.videos (
    source_db text not null,
    video_id text not null,
    video_url text not null,
    video_title text,
    description text,
    game text not null default 'lol',
    role text not null,
    subject text,
    champion text,
    rank text,
    website_rating double precision,
    message_timestamp text,
    status text,
    transcription text,
    source text default 'discord',
    created_at timestamptz default now(),
    primary key (source_db, video_id)
);

create table if not exists public.insights (
    source_db text not null,
    local_id bigint not null,
    video_id text not null,
    insight_type text not null,
    text text not null,
    subject text,
    subject_type text,
    source_score double precision,
    cluster_score double precision,
    confidence double precision,
    repetition_count integer default 1,
    is_duplicate boolean default false,
    embedding vector(384),
    created_at timestamptz default now(),
    primary key (source_db, local_id),
    foreign key (source_db, video_id) references public.videos(source_db, video_id)
);

create table if not exists public.runtime_config (
    key text primary key,
    value jsonb not null,
    updated_at timestamptz default now()
);

create index if not exists videos_game_idx on public.videos (game);
create index if not exists videos_source_idx on public.videos (source);
create index if not exists videos_subject_idx on public.videos (lower(coalesce(subject, champion)));
create index if not exists videos_champion_idx on public.videos (lower(champion));
create index if not exists videos_role_idx on public.videos (role);
create index if not exists insights_type_idx on public.insights (insight_type);
create index if not exists insights_subject_idx on public.insights (lower(subject));
create index if not exists insights_embedding_hnsw_idx
    on public.insights using hnsw (embedding vector_cosine_ops)
    where embedding is not null;

alter table public.runtime_config enable row level security;

drop policy if exists runtime_config_public_read on public.runtime_config;
create policy runtime_config_public_read
    on public.runtime_config
    for select
    to anon, authenticated
    using (true);

create or replace function public.match_insights(
    query_embedding vector(384),
    match_count integer,
    p_game text default null,
    p_role text default null,
    p_champion text default null,
    p_subject text default null,
    p_insight_type text default null,
    p_general_aoe2_only boolean default false
)
returns table (
    source_db text,
    local_id bigint,
    video_id text,
    text text,
    insight_type text,
    role text,
    subject text,
    insight_subject text,
    subject_type text,
    champion text,
    game text,
    rank text,
    website_rating double precision,
    source text,
    confidence double precision,
    source_score double precision,
    score double precision
)
language sql
stable
as $$
    select
        i.source_db,
        i.local_id,
        i.video_id,
        i.text,
        i.insight_type,
        v.role,
        case
            when i.subject_type is not null then i.subject
            else coalesce(v.subject, v.champion)
        end as subject,
        i.subject as insight_subject,
        i.subject_type,
        v.champion,
        v.game,
        v.rank,
        v.website_rating,
        coalesce(v.source, 'discord') as source,
        i.confidence,
        i.source_score,
        1 - (i.embedding <=> query_embedding) as score
    from public.insights i
    join public.videos v
      on v.source_db = i.source_db
     and v.video_id = i.video_id
    where i.embedding is not null
      and (p_game is null or v.game = p_game)
      and (p_role is null or v.role = p_role)
      and (p_champion is null or lower(v.champion) = lower(p_champion))
      and (
          p_subject is null
          or lower(case
              when i.subject_type is not null then i.subject
              else coalesce(v.subject, v.champion)
          end) = lower(p_subject)
      )
      and (p_insight_type is null or i.insight_type = p_insight_type)
      and (
          not p_general_aoe2_only
          or coalesce(i.subject_type, 'general') = 'general'
      )
    order by i.embedding <=> query_embedding
    limit match_count;
$$;
