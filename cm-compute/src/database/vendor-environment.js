class VendorEnvironment {
  constructor(pool) {
    this.pool = pool;
  }

  async create({ name, description = '', installations = [] }) {
    const result = await this.pool.query(
      `INSERT INTO vendor_environments (name, description, installations)
       VALUES ($1, $2, $3)
       RETURNING *`,
      [name, description, JSON.stringify(installations)]
    );
    return this._parseRow(result.rows[0]);
  }

  async findAll() {
    const result = await this.pool.query(
      `SELECT * FROM vendor_environments ORDER BY name`
    );
    return result.rows.map(row => this._parseRow(row));
  }

  async findById(id) {
    const result = await this.pool.query(
      `SELECT * FROM vendor_environments WHERE id = $1`,
      [id]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async findByName(name) {
    const result = await this.pool.query(
      `SELECT * FROM vendor_environments WHERE name = $1`,
      [name]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async update(id, fields) {
    const allowedFields = ['name', 'description', 'installations'];
    const updates = [];
    const values = [];
    let paramIndex = 1;

    for (const [key, value] of Object.entries(fields)) {
      if (allowedFields.includes(key)) {
        updates.push(`${key} = $${paramIndex}`);
        values.push(key === 'installations' ? JSON.stringify(value) : value);
        paramIndex++;
      }
    }

    if (updates.length === 0) {
      return this.findById(id);
    }

    values.push(id);
    const result = await this.pool.query(
      `UPDATE vendor_environments
       SET ${updates.join(', ')}
       WHERE id = $${paramIndex}
       RETURNING *`,
      values
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async addInstallation(id, installation) {
    // installation: { repo, branch, build_type, install_prefix, cmake_options }
    const env = await this.findById(id);
    if (!env) return null;

    const installations = [...env.installations, installation];
    return this.update(id, { installations });
  }

  async removeInstallation(id, repo) {
    const env = await this.findById(id);
    if (!env) return null;

    const installations = env.installations.filter(i => i.repo !== repo);
    return this.update(id, { installations });
  }

  async delete(id) {
    const result = await this.pool.query(
      `DELETE FROM vendor_environments WHERE id = $1 RETURNING *`,
      [id]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async deleteByName(name) {
    const result = await this.pool.query(
      `DELETE FROM vendor_environments WHERE name = $1 RETURNING *`,
      [name]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async getLinkedCppEnvironments(vendorEnvId) {
    const result = await this.pool.query(
      `SELECT ce.* FROM cpp_environments ce
       JOIN cpp_vendor_links cvl ON ce.id = cvl.cpp_env_id
       WHERE cvl.vendor_env_id = $1
       ORDER BY ce.name`,
      [vendorEnvId]
    );
    return result.rows.map(row => ({
      ...row,
      packages: typeof row.packages === 'string'
        ? JSON.parse(row.packages)
        : row.packages
    }));
  }

  _parseRow(row) {
    return {
      ...row,
      installations: typeof row.installations === 'string'
        ? JSON.parse(row.installations)
        : row.installations
    };
  }
}

module.exports = VendorEnvironment;
