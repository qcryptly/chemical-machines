-- C++ Environment Management Schema
-- Run this migration to set up the C++ environment tables

-- C++ environments (system packages via apt)
CREATE TABLE IF NOT EXISTS cpp_environments (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) UNIQUE NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  packages JSONB DEFAULT '[]'::jsonb  -- Array of installed debian packages
);

-- Vendor environments (source installations from git repos)
CREATE TABLE IF NOT EXISTS vendor_environments (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) UNIQUE NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  installations JSONB DEFAULT '[]'::jsonb  -- Array of {repo, branch, build_type, install_prefix, cmake_options}
);

-- Link C++ environments to vendor environments (many-to-many)
CREATE TABLE IF NOT EXISTS cpp_vendor_links (
  cpp_env_id INTEGER REFERENCES cpp_environments(id) ON DELETE CASCADE,
  vendor_env_id INTEGER REFERENCES vendor_environments(id) ON DELETE CASCADE,
  created_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (cpp_env_id, vendor_env_id)
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_cpp_environments_name ON cpp_environments(name);
CREATE INDEX IF NOT EXISTS idx_vendor_environments_name ON vendor_environments(name);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_cpp_environments_updated_at ON cpp_environments;
CREATE TRIGGER update_cpp_environments_updated_at
    BEFORE UPDATE ON cpp_environments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_vendor_environments_updated_at ON vendor_environments;
CREATE TRIGGER update_vendor_environments_updated_at
    BEFORE UPDATE ON vendor_environments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
