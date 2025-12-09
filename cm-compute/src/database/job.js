class Job {
  constructor(pool) {
    this.pool = pool;
  }

  async create({ type, params, priority = 5, status = 'queued' }) {
    const result = await this.pool.query(
      `INSERT INTO compute_jobs (type, params, priority, status, created_at)
       VALUES ($1, $2, $3, $4, NOW())
       RETURNING *`,
      [type, JSON.stringify(params), priority, status]
    );
    return this._parseRow(result.rows[0]);
  }

  async findById(id) {
    const result = await this.pool.query(
      `SELECT * FROM compute_jobs WHERE id = $1 AND deleted_at IS NULL`,
      [id]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async findByStatus(status, limit = 100) {
    const result = await this.pool.query(
      `SELECT * FROM compute_jobs
       WHERE status = $1 AND deleted_at IS NULL
       ORDER BY priority DESC, created_at ASC
       LIMIT $2`,
      [status, limit]
    );
    return result.rows.map(row => this._parseRow(row));
  }

  async update(id, fields) {
    const allowedFields = ['type', 'params', 'priority', 'status', 'result', 'started_at', 'completed_at'];
    const updates = [];
    const values = [];
    let paramIndex = 1;

    for (const [key, value] of Object.entries(fields)) {
      if (allowedFields.includes(key)) {
        updates.push(`${key} = $${paramIndex}`);
        values.push(key === 'params' || key === 'result' ? JSON.stringify(value) : value);
        paramIndex++;
      }
    }

    if (updates.length === 0) {
      return this.findById(id);
    }

    values.push(id);
    const result = await this.pool.query(
      `UPDATE compute_jobs
       SET ${updates.join(', ')}
       WHERE id = $${paramIndex} AND deleted_at IS NULL
       RETURNING *`,
      values
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async updateStatus(id, status, result = null) {
    const fields = { status };

    if (status === 'running') {
      fields.started_at = new Date().toISOString();
    } else if (status === 'completed' || status === 'failed') {
      fields.completed_at = new Date().toISOString();
      if (result !== null) {
        fields.result = result;
      }
    }

    return this.update(id, fields);
  }

  async softDelete(id) {
    const result = await this.pool.query(
      `UPDATE compute_jobs
       SET deleted_at = NOW()
       WHERE id = $1 AND deleted_at IS NULL
       RETURNING *`,
      [id]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async delete(id) {
    const result = await this.pool.query(
      `DELETE FROM compute_jobs WHERE id = $1 RETURNING *`,
      [id]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  async restore(id) {
    const result = await this.pool.query(
      `UPDATE compute_jobs
       SET deleted_at = NULL
       WHERE id = $1
       RETURNING *`,
      [id]
    );
    return result.rows[0] ? this._parseRow(result.rows[0]) : null;
  }

  _parseRow(row) {
    return {
      ...row,
      id: BigInt(row.id),
      params: typeof row.params === 'string' ? JSON.parse(row.params) : row.params,
      result: typeof row.result === 'string' ? JSON.parse(row.result) : row.result
    };
  }
}

module.exports = Job;
