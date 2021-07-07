//Show classification buttons once video has loaded
$(document).ready(function(){
	$.ajaxSetup({ cache: false });
     // or iPhones don't get fresh data
});

$("#video-feed").on('load', function(){
    var today = new Date();
    $.ajax({
        url: '/dummyajax',
        data: today,
        type: 'POST',
        success: function(response){
            console.log(response);
        },
        error: function(error){
            console.log(error);
        }
    });
    $('.loader').addClass('d-none');
    $('.btn-group').removeClass('d-none');
    $('.btn-group').addClass('d-block');
    $('.bk').removeClass('d-none');
});

//Send classification data point to be saved from live stream
$(function(){
    $('button.live-classification').click(function(){
        value = $(this).attr('value');
        $.ajax({
            url: '/classify',
            data: value,
            type: 'POST',
            success: function(response){
                $(".alert-container").prepend("<div class='alert alert-success alert-dismissible fade show' role='alert'><strong>Success! </strong>" + response + "<button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button></div>");
                $(".alert").delay(3000).slideUp(200, function() {
                    $(this).removeClass('show');
                });
            },
            error: function(error){
                console.log(error);
            }
        });
    });
});

//Send classification data point to be saved
$(function(){
    $('button.correction').click(function(){
        $t = $(this);
        new_value = $(this).attr('value')

        data_to_send = {
            id: $(this).closest('tr').children('td.id-column').text(),
            value: new_value
        };
        $.ajax({
            url: '/correct',
            contentType: "application/json",
            data: JSON.stringify(data_to_send),
            type: 'POST',
            success: function(response){
                if ( new_value == 'Empty'){
                    old_value = 'Baby';
                } else {
                    old_value = 'Empty';
                }
                $t.val(old_value);
                $t.text(old_value);
                $t.closest('div').siblings('button.dropdown-toggle').text(new_value);

                console.log(response);
            },
            error: function(error){
                console.log(error);
            }
        });
    });
});

$('button.reading-classification').click(function(){
    $t = $(this);
    value = $(this).attr('value');
    id = $(this).closest('tr').children('td.id-column').text();
    
    data_to_send = {
        id: id,
        value: value,
    };
    $.ajax({
        url: '/classifyreading',
        contentType: "application/json",
        data: JSON.stringify(data_to_send),
        type: 'POST',
        success: function(response){
            $t.closest('div').addClass('d-none');
            $t.closest('tr').find('div.included-in').removeClass('d-none');
            console.log(response);
        },
        error: function(error){
            console.log(error);
        }
    });
    
});

$('#playlivetext').click(function(){
    $('#myModal').modal();
});