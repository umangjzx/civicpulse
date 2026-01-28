// Main JavaScript for CivicPulse

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Auto-dismiss alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
    
    // Initialize notifications
    initializeNotifications();
});

// ==================== NOTIFICATION SYSTEM ====================

function initializeNotifications() {
    // Check if user is logged in and notification elements exist
    const notificationToggle = document.getElementById('notificationToggle');
    if (!notificationToggle) return;
    
    // Load notifications on page load
    loadNotifications();
    
    // Refresh notifications every 10 seconds
    setInterval(loadNotifications, 10000);
    
    // Mark all as read button
    const markAllReadBtn = document.getElementById('markAllReadBtn');
    if (markAllReadBtn) {
        markAllReadBtn.addEventListener('click', markAllNotificationsRead);
    }
    
    // Refresh notifications when offcanvas is opened
    const notificationOffcanvas = document.getElementById('notificationOffcanvas');
    if (notificationOffcanvas) {
        notificationOffcanvas.addEventListener('show.bs.offcanvas', loadNotifications);
    }
}

function loadNotifications() {
    fetch('/api/notifications?limit=50')
        .then(response => response.json())
        .then(data => {
            displayNotifications(data.notifications);
            updateUnreadCount();
        })
        .catch(error => console.error('Error loading notifications:', error));
}

function displayNotifications(notifications) {
    const notificationsList = document.getElementById('notificationsList');
    if (!notificationsList) return;
    
    if (notifications.length === 0) {
        notificationsList.innerHTML = '<div class="text-center p-4"><p class="text-muted"><i class="fas fa-check-circle"></i> No notifications</p></div>';
        return;
    }
    
    let html = '<div class="list-group list-group-flush">';
    
    notifications.forEach(notif => {
        const timeAgo = formatTimeAgo(notif.created_at);
        const badgeColor = notif.type === 'error' ? 'danger' : notif.type === 'warning' ? 'warning' : notif.type === 'success' ? 'success' : 'info';
        const unreadClass = notif.is_read ? '' : 'bg-light';
        
        html += `
            <div class="list-group-item ${unreadClass} border-start border-4 border-${badgeColor}" data-notification-id="${notif.id}">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <h6 class="mb-1">
                            <span class="badge bg-${badgeColor} me-2">${notif.type.toUpperCase()}</span>
                            ${notif.title}
                        </h6>
                        <p class="mb-2 small">${notif.message}</p>
                        <small class="text-muted">${timeAgo}</small>
                    </div>
                    <div class="dropdown">
                        <button class="btn btn-sm btn-link dropdown-toggle" type="button" data-bs-toggle="dropdown">
                            <i class="fas fa-ellipsis-v"></i>
                        </button>
                        <ul class="dropdown-menu dropdown-menu-end">
                            ${!notif.is_read ? `<li><a class="dropdown-item" onclick="markNotificationRead(${notif.id})"><i class="fas fa-check"></i> Mark as read</a></li>` : ''}
                            <li><a class="dropdown-item text-danger" onclick="deleteNotification(${notif.id})"><i class="fas fa-trash"></i> Delete</a></li>
                        </ul>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    notificationsList.innerHTML = html;
}

function updateUnreadCount() {
    fetch('/api/notifications/unread-count')
        .then(response => response.json())
        .then(data => {
            const badge = document.getElementById('notificationBadge');
            const unreadCount = document.getElementById('unreadCount');
            
            if (data.unread_count > 0) {
                badge.style.display = 'inline-block';
                unreadCount.textContent = data.unread_count;
            } else {
                badge.style.display = 'none';
            }
        })
        .catch(error => console.error('Error updating unread count:', error));
}

function markNotificationRead(notificationId) {
    fetch(`/api/notifications/${notificationId}/read`, {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            loadNotifications();
        })
        .catch(error => console.error('Error marking notification as read:', error));
}

function markAllNotificationsRead() {
    fetch('/api/notifications/mark-all-read', {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            loadNotifications();
            showToast('All notifications marked as read', 'success');
        })
        .catch(error => console.error('Error marking all as read:', error));
}

function deleteNotification(notificationId) {
    if (confirm('Delete this notification?')) {
        fetch(`/api/notifications/${notificationId}`, {method: 'DELETE'})
            .then(response => response.json())
            .then(data => {
                loadNotifications();
            })
            .catch(error => console.error('Error deleting notification:', error));
    }
}

function formatTimeAgo(timestamp) {
    const now = new Date();
    const date = new Date(timestamp);
    const seconds = Math.floor((now - date) / 1000);
    
    if (seconds < 60) return 'Just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

function showToast(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(alertDiv);
    
    setTimeout(() => alertDiv.remove(), 4000);
}

// Geolocation for complaint submission
function getLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                document.getElementById('latitude').value = position.coords.latitude;
                document.getElementById('longitude').value = position.coords.longitude;
                
                // Reverse geocode to get address (simplified)
                fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${position.coords.latitude}&lon=${position.coords.longitude}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.display_name) {
                            document.getElementById('location').value = data.display_name;
                        }
                    })
                    .catch(error => console.error('Geocoding error:', error));
            },
            function(error) {
                console.error('Geolocation error:', error);
                alert('Unable to get location. Please enter manually.');
            }
        );
    } else {
        alert('Geolocation is not supported by your browser.');
    }
}

// Preview image before upload
function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        const preview = document.getElementById('imagePreview');
        
        reader.onload = function(e) {
            if (!preview) {
                const container = input.parentNode;
                const img = document.createElement('img');
                img.id = 'imagePreview';
                img.src = e.target.result;
                img.className = 'img-thumbnail mt-2';
                img.style.maxHeight = '200px';
                container.appendChild(img);
            } else {
                preview.src = e.target.result;
            }
        };
        
        reader.readAsDataURL(input.files[0]);
    }
}

// Upvote complaint
function upvoteComplaint(complaintId) {
    fetch(`/complaint/${complaintId}/upvote`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const upvoteBtn = document.getElementById(`upvoteBtn${complaintId}`);
                const upvoteCount = document.getElementById(`upvoteCount${complaintId}`);
                
                upvoteBtn.disabled = true;
                upvoteBtn.classList.add('btn-success');
                upvoteBtn.classList.remove('btn-outline-primary');
                upvoteBtn.innerHTML = '<i class="fas fa-check"></i> Upvoted';
                
                upvoteCount.textContent = parseInt(upvoteCount.textContent) + 1;
                
                showToast('Upvote recorded!', 'success');
            } else {
                showToast(data.error || 'Already upvoted', 'warning');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error upvoting complaint', 'danger');
        });
}

// Show toast notification
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(container);
    }
    
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `toast align-items-center text-bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.getElementById('toastContainer').appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

// Status update for admin
function updateComplaintStatus(complaintId, status) {
    const notes = prompt('Add notes (optional):');
    if (notes === null) return; // User cancelled
    
    fetch(`/admin/complaint/${complaintId}/update`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `status=${status}&notes=${encodeURIComponent(notes || '')}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Status updated successfully', 'success');
            setTimeout(() => location.reload(), 1000);
        } else {
            showToast(data.error || 'Update failed', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showToast('Error updating status', 'danger');
    });
}

// Initialize complaint map
function initComplaintMap(complaints) {
    if (typeof L === 'undefined') return;
    
    const map = L.map('complaintMap').setView([20.5937, 78.9629], 5);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    complaints.forEach(complaint => {
        if (complaint.latitude && complaint.longitude) {
            const marker = L.marker([complaint.latitude, complaint.longitude]).addTo(map);
            marker.bindPopup(`
                <strong>${complaint.title}</strong><br>
                <small>Category: ${complaint.category}<br>
                Status: ${complaint.status}<br>
                <a href="/complaint/${complaint.id}" target="_blank">View Details</a></small>
            `);
        }
    });
}

// Form validation
function validateComplaintForm() {
    const title = document.getElementById('title').value.trim();
    const description = document.getElementById('description').value.trim();
    const category = document.getElementById('category').value;
    const location = document.getElementById('location').value.trim();
    
    if (!title || !description || !category || !location) {
        showToast('Please fill in all required fields', 'warning');
        return false;
    }
    
    if (description.length < 20) {
        showToast('Description should be at least 20 characters', 'warning');
        return false;
    }
    
    return true;
}

// Export data
function exportData(format = 'csv') {
    showToast(`Exporting data as ${format.toUpperCase()}...`, 'info');
    // In a real implementation, this would make an API call to export data
    setTimeout(() => {
        showToast('Export complete!', 'success');
    }, 2000);
}

// Search functionality
function searchComplaints() {
    const query = document.getElementById('searchInput').value.toLowerCase();
    const complaints = document.querySelectorAll('.complaint-card');
    
    complaints.forEach(card => {
        const text = card.textContent.toLowerCase();
        card.style.display = text.includes(query) ? '' : 'none';
    });
}

// Auto-refresh dashboard (every 30 seconds)
if (window.location.pathname.includes('dashboard') || 
    window.location.pathname.includes('admin')) {
    setInterval(() => {
        // Refresh data without reloading page
        fetch('/api/dashboard-stats')
            .then(response => response.json())
            .then(data => {
                // Update dashboard stats if elements exist
                const elements = {
                    'totalComplaints': data.total_complaints,
                    'resolvedComplaints': data.resolved,
                    'inProgressComplaints': data.in_progress,
                    'pendingComplaints': data.pending
                };
                
                for (const [id, value] of Object.entries(elements)) {
                    const element = document.getElementById(id);
                    if (element) {
                        element.textContent = value;
                    }
                }
            });
    }, 30000);
}