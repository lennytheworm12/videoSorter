import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey =
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ??
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
export const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const hasSupabaseConfig = Boolean(supabaseUrl && supabaseAnonKey);

export const supabase = hasSupabaseConfig
  ? createClient(supabaseUrl!, supabaseAnonKey!, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
        flowType: "pkce",
        storageKey: "videosorter-auth",
      },
    })
  : null;

export async function fetchRuntimeConfig<T>(key: string): Promise<T | null> {
  if (!supabase) return null;
  const { data, error } = await supabase
    .from("runtime_config")
    .select("value")
    .eq("key", key)
    .maybeSingle();
  if (error) {
    throw error;
  }
  return (data?.value as T | undefined) ?? null;
}
