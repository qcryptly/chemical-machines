class CppEnvironment {
  constructor(pool) {
    this.pool = pool;
  }

  async create({ name, description = '', packages = [] }) {
    const result = await this.pool.query(
      `INSERT INTO cpp_environments (name, description, packages)
       VALUES ($1, $2, $3)
       RETURNING *`,
      [name, description, JSON.stringify(packages)]
    );
    return this._parseRow(result.rows[0]);
  }

  async findAll() {
    const result = await this.pool.query(
      `SELECT * FROM cpp_environments ORDER BY name`
    );
    return result.rows.map(row => this._parseRow(row));
  }

  async findById(id) {
    const result = await this.pool.query(
      `SELECT * FROM cpp_environments WHERE id = $1`,
      [id]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async findByName(name) {
    const result = await this.pool.query(
      `SELECT * FROM cpp_environments WHERE name = $1`,
      [name]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async update(id, fields) {
    const allowedFields = ['name', 'description', 'packages'];
    const updates = [];
    const values = [];
    let paramIndex = 1;

    for (const [key, value] of Object.entries(fields)) {
      if (allowedFields.includes(key)) {
        updates.push(`${key} = $${paramIndex}`);
        values.push(key === 'packages' ? JSON.stringify(value) : value);
        paramIndex++;
      }
    }

    if (updates.length === 0) {
      return this.findById(id);
    }

    values.push(id);
    const result = await this.pool.query(
      `UPDATE cpp_environments
       SET ${updates.join(', ')}
       WHERE id = $${paramIndex}
       RETURNING *`,
      values
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async addPackages(id, newPackages) {
    const env = await this.findById(id);
    if (!env) return null;

    const packages = [...new Set([...env.packages, ...newPackages])];
    return this.update(id, { packages });
  }

  async delete(id) {
    const result = await this.pool.query(
      `DELETE FROM cpp_environments WHERE id = $1 RETURNING *`,
      [id]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async deleteByName(name) {
    const result = await this.pool.query(
      `DELETE FROM cpp_environments WHERE name = $1 RETURNING *`,
      [name]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async linkVendorEnvironment(cppEnvId, vendorEnvId) {
    await this.pool.query(
      `INSERT INTO cpp_vendor_links (cpp_env_id, vendor_env_id)
       VALUES ($1, $2)
       ON CONFLICT DO NOTHING`,
      [cppEnvId, vendorEnvId]
    );
  }

  async unlinkVendorEnvironment(cppEnvId, vendorEnvId) {
    await this.pool.query(
      `DELETE FROM cpp_vendor_links
       WHERE cpp_env_id = $1 AND vendor_env_id = $2`,
      [cppEnvId, vendorEnvId]
    );
  }

  async getLinkedVendorEnvironments(cppEnvId) {
    const result = await this.pool.query(
      `SELECT ve.* FROM vendor_environments ve
       JOIN cpp_vendor_links cvl ON ve.id = cvl.vendor_env_id
       WHERE cvl.cpp_env_id = $1
       ORDER BY ve.name`,
      [cppEnvId]
    );
    return result.rows.map(row => ({
      ...row,
      installations: typeof row.installations === 'string'
        ? JSON.parse(row.installations)
        : row.installations
    }));
  }

  _parseRow(row) {
    return {
      ...row,
      packages: typeof row.packages === 'string'
        ? JSON.parse(row.packages)
        : row.packages
    };
  }
}

module.exports = CppEnvironment;
