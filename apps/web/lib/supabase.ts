import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey =
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ??
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const hasSupabaseConfig = Boolean(supabaseUrl && supabaseAnonKey);

export const supabase = hasSupabaseConfig
  ? createClient(supabaseUrl!, supabaseAnonKey!)
  : null;
